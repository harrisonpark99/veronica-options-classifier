# ============================================
# SSL 문제 해결 + 전략 데이터 스크래퍼 (DateRange 입력 Robust 버전)
# - 로그인/리다이렉트 상황에서도 안전하게 대기
# - DateRange input 셀렉터 다중 시도 + iframe 탐색
# - 실패 시 input 목록 덤프로 디버깅 가능
# - [개선] 날짜 입력 후 실제 값 확인 + 로그 출력
# ============================================

import os
import certifi

# --- SSL 인증서 문제 우회(프로세스 전역) ---
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

if "REQUESTS_CA_BUNDLE" in os.environ and "/opt/homebrew" in os.environ["REQUESTS_CA_BUNDLE"]:
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

print(f"SSL 인증서 경로 설정됨: {certifi.where()}")

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from datetime import datetime, timedelta
import pandas as pd
import time

# webdriver-manager는 SSL 설정 후 import
try:
    from webdriver_manager.chrome import ChromeDriverManager
except Exception:
    print("webdriver-manager 설치 중...")
    import subprocess
    import sys

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--trusted-host",
            "pypi.org",
            "--trusted-host",
            "files.pythonhosted.org",
            "webdriver-manager",
        ]
    )
    from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://xunkemgmt-analytics-2.joomo.io/analytics/snt/pta/strategies"


def dump_inputs_for_debug(driver, label=""):
    """현재 DOM에 존재하는 input들을 최대한 덤프해서 셀렉터를 확정할 수 있게 해줌"""
    try:
        data = driver.execute_script(
            r"""
        const inputs = Array.from(document.querySelectorAll('input'));
        return inputs.slice(0, 80).map((el) => ({
          tag: el.tagName,
          type: el.getAttribute('type'),
          id: el.getAttribute('id'),
          name: el.getAttribute('name'),
          placeholder: el.getAttribute('placeholder'),
          value: el.value,
          dataRange: el.getAttribute('data-range'),
          ariaLabel: el.getAttribute('aria-label'),
          class: el.getAttribute('class')
        }));
        """
        )
        print(f"\n[DEBUG] Input dump {label} (top {len(data)}):")
        for i, row in enumerate(data[:30]):
            print(f"  #{i+1}: {row}")
    except Exception as e:
        print(f"[DEBUG] input dump failed: {e}")


def is_login_like_page(driver) -> bool:
    """로그인/인증 페이지로 보이는지 대충 휴리스틱 체크"""
    try:
        url = driver.current_url.lower()
        title = (driver.title or "").lower()
        if any(k in url for k in ["login", "signin", "auth", "sso", "oauth", "okta", "azure", "microsoftonline"]):
            return True
        if any(k in title for k in ["login", "sign in", "authentication", "authorize"]):
            return True
        return False
    except Exception:
        return False


def wait_for_user_login(driver, timeout=180):
    """
    사용자가 수동 로그인할 시간을 주고,
    로그인 이후 BASE_URL로 돌아오거나, timeRange input이 나타나면 진행.
    """
    print("\n⚠️ 로그인이 필요하면 지금 Selenium 크롬에서 로그인해주세요.")
    print(f"최대 {timeout}초 대기합니다...\n")

    start = time.time()
    last_print = 0

    while time.time() - start < timeout:
        # 주기적으로 상태 출력
        if time.time() - last_print > 10:
            print(f"  ...대기 중 (현재 URL: {driver.current_url})")
            last_print = time.time()

        # 로그인 페이지를 벗어났는지 + 날짜 입력이 등장했는지 확인
        try:
            driver.switch_to.default_content()
            # quick check selectors
            found = driver.execute_script(
                r"""
            return !!(
              document.querySelector('input[data-range="start"]') ||
              document.querySelector('input[data-range="end"]') ||
              document.querySelector('#timeRange') ||
              document.querySelector('input[placeholder*="Start"]') ||
              document.querySelector('input[placeholder*="End"]')
            );
            """
            )
            if found and (BASE_URL.split("/analytics/")[0] in driver.current_url):
                return True
            if found:
                return True
            if (not is_login_like_page(driver)) and (BASE_URL in driver.current_url):
                return True
        except Exception:
            pass

        time.sleep(1)

    print("⚠️ 로그인 대기 시간이 끝났습니다. (계속 진행 시 실패할 수 있어요)")
    return False


def _get_input_value(driver, css_selector: str) -> str | None:
    """
    css_selector로 input을 찾아서 현재 값을 반환.
    못 찾으면 None 반환.
    """
    try:
        value = driver.execute_script(
            r"""
            const el = document.querySelector(arguments[0]);
            if (!el) return null;
            return el.value;
            """,
            css_selector,
        )
        return value
    except Exception:
        return None


def _set_react_input_value(driver, css_selector: str, value: str) -> bool:
    """
    css_selector로 element 찾고 React valueTracker 고려하여 값 주입.
    찾았으면 True, 못 찾으면 False.
    """
    try:
        ok = driver.execute_script(
            r"""
        function setReactInput(el, value) {
          const lastValue = el.value;
          el.value = value;

          const tracker = el._valueTracker;
          if (tracker) tracker.setValue(lastValue);

          el.dispatchEvent(new Event('input', { bubbles: true }));
          el.dispatchEvent(new Event('change', { bubbles: true }));
        }

        const el = document.querySelector(arguments[0]);
        if (!el) return false;
        setReactInput(el, arguments[1]);
        return true;
        """,
            css_selector,
            value,
        )
        return bool(ok)
    except Exception:
        return False


def _verify_input_value(driver, css_selector: str, expected_value: str) -> tuple[bool, str | None]:
    """
    입력 후 실제로 값이 들어갔는지 확인.

    Returns:
        (성공여부, 실제값)
    """
    actual_value = _get_input_value(driver, css_selector)
    if actual_value is None:
        return False, None

    # 값이 일치하는지 확인
    is_match = actual_value.strip() == expected_value.strip()
    return is_match, actual_value


def set_time_range_robust(driver, start_date: str, end_date: str, timeout=25) -> dict:
    """
    - default DOM에서 시도
    - 실패하면 iframe 내부까지 순회하며 시도
    - 여러 셀렉터 후보를 사용
    - 끝까지 실패하면 input 덤프 + 에러

    Returns:
        dict: {
            'success': bool,
            'start_requested': str,  # 요청한 시작 날짜
            'end_requested': str,    # 요청한 종료 날짜
            'start_actual': str,     # 실제 입력된 시작 날짜
            'end_actual': str,       # 실제 입력된 종료 날짜
            'start_selector': str,   # 사용된 시작 셀렉터
            'end_selector': str,     # 사용된 종료 셀렉터
        }
    """
    result = {
        'success': False,
        'start_requested': start_date,
        'end_requested': end_date,
        'start_actual': None,
        'end_actual': None,
        'start_selector': None,
        'end_selector': None,
    }

    selectors_start = [
        'input[data-range="start"]',
        '#timeRange',  # 캡처에서 start input에 id=timeRange가 있었음
        'input[placeholder="Start date"]',
        'input[placeholder*="Start"]',
        'input[aria-label*="Start"]',
    ]
    selectors_end = [
        'input[data-range="end"]',
        'input[placeholder="End date"]',
        'input[placeholder*="End"]',
        'input[aria-label*="End"]',
    ]

    def try_in_current_context() -> bool:
        nonlocal result

        # === 입력 전 현재 값 확인 ===
        print("\n[DEBUG] 입력 전 현재 값 확인:")
        for sel in selectors_start + selectors_end:
            current_val = _get_input_value(driver, sel)
            if current_val is not None:
                print(f"  {sel}: '{current_val}'")

        # === Start 날짜 입력 ===
        start_ok = False
        for sel in selectors_start:
            if _set_react_input_value(driver, sel, start_date):
                # 입력 후 실제 값 확인
                time.sleep(0.3)  # React 반영 대기
                is_match, actual = _verify_input_value(driver, sel, start_date)

                if is_match:
                    result['start_actual'] = actual
                    result['start_selector'] = sel
                    start_ok = True
                    print(f"  ✅ Start 입력 성공: {sel} = '{actual}'")
                    break
                else:
                    print(f"  ⚠️ Start 입력 시도했으나 값 불일치: {sel}")
                    print(f"      요청: '{start_date}' → 실제: '{actual}'")
                    # 값이 다르더라도 일단 기록
                    if actual:
                        result['start_actual'] = actual
                        result['start_selector'] = sel

        # === End 날짜 입력 ===
        end_ok = False
        for sel in selectors_end:
            if _set_react_input_value(driver, sel, end_date):
                # 입력 후 실제 값 확인
                time.sleep(0.3)  # React 반영 대기
                is_match, actual = _verify_input_value(driver, sel, end_date)

                if is_match:
                    result['end_actual'] = actual
                    result['end_selector'] = sel
                    end_ok = True
                    print(f"  ✅ End 입력 성공: {sel} = '{actual}'")
                    break
                else:
                    print(f"  ⚠️ End 입력 시도했으나 값 불일치: {sel}")
                    print(f"      요청: '{end_date}' → 실제: '{actual}'")
                    # 값이 다르더라도 일단 기록
                    if actual:
                        result['end_actual'] = actual
                        result['end_selector'] = sel

        result['success'] = start_ok and end_ok
        return start_ok and end_ok

    # 1) 기본 컨텍스트에서 timeout 동안 반복 시도 (렌더 지연 대비)
    t0 = time.time()
    driver.switch_to.default_content()
    while time.time() - t0 < timeout:
        if try_in_current_context():
            return result
        time.sleep(0.5)

    # 2) iframe 탐색
    driver.switch_to.default_content()
    frames = driver.find_elements(By.TAG_NAME, "iframe")
    if frames:
        print(f"[INFO] iframe {len(frames)}개 감지 → iframe 내부도 탐색합니다.")
    for i, fr in enumerate(frames):
        try:
            driver.switch_to.default_content()
            driver.switch_to.frame(fr)
            t1 = time.time()
            while time.time() - t1 < 5:
                if try_in_current_context():
                    driver.switch_to.default_content()
                    return result
                time.sleep(0.5)
        except Exception:
            continue

    # 3) 실패 시 디버깅 출력
    driver.switch_to.default_content()
    dump_inputs_for_debug(driver, label="(default content)")
    # iframe 안 input도 덤프 시도
    for i, fr in enumerate(frames[:3]):  # 너무 많을 수 있어 상위 3개만
        try:
            driver.switch_to.default_content()
            driver.switch_to.frame(fr)
            dump_inputs_for_debug(driver, label=f"(iframe {i})")
        except Exception:
            pass

    driver.switch_to.default_content()
    return result


def print_date_summary(result: dict):
    """날짜 입력 결과를 보기 좋게 출력"""
    print("\n" + "=" * 50)
    print("📅 날짜 입력 결과 요약")
    print("=" * 50)

    print(f"\n[요청한 날짜]")
    print(f"  FROM: {result['start_requested']}")
    print(f"  TO:   {result['end_requested']}")

    print(f"\n[실제 입력된 날짜]")
    print(f"  FROM: {result['start_actual'] or '❌ 입력 실패'}")
    print(f"  TO:   {result['end_actual'] or '❌ 입력 실패'}")

    # 날짜 일치 여부 확인
    start_match = result['start_requested'] == result['start_actual']
    end_match = result['end_requested'] == result['end_actual']

    print(f"\n[검증 결과]")
    if start_match and end_match:
        print("  ✅ 모든 날짜가 정확히 입력되었습니다!")
    else:
        if not start_match:
            print(f"  ⚠️ FROM 날짜 불일치!")
            print(f"      요청: {result['start_requested']} → 실제: {result['start_actual']}")
        if not end_match:
            print(f"  ⚠️ TO 날짜 불일치!")
            print(f"      요청: {result['end_requested']} → 실제: {result['end_actual']}")

    print("=" * 50 + "\n")

    return start_match and end_match


def click_submit(driver):
    buttons = driver.find_elements(By.TAG_NAME, "button")
    for btn in buttons:
        if btn.text.strip().lower() == "submit":
            btn.click()
            return
    raise Exception("Submit 버튼을 찾지 못했습니다.")


def wait_for_table(driver, timeout=25):
    wait = WebDriverWait(driver, timeout)
    wait.until(EC.presence_of_element_located((By.TAG_NAME, "table")))


def scrape_strategy_data(start_date, end_date, headless=False):
    print(f"\n{'='*60}")
    print(f"스크래핑 시작")
    print(f"  요청 날짜: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    options = webdriver.ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--ignore-certificate-errors")
    options.add_argument("--ignore-ssl-errors")

    print("크롬 드라이버 다운로드 및 설정 중...")

    try:
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        print(f"ChromeDriverManager 자동 설치 실패: {e}")
        print("ChromeDriver를 시스템에서 찾도록 시도합니다...")
        driver = webdriver.Chrome(options=options)

    driver.set_window_size(1400, 900)

    try:
        print(f"페이지 접속: {BASE_URL}")
        driver.get(BASE_URL)
        time.sleep(2)

        # 로그인 페이지면 기다리기 (로그인 로직은 나중에 하기로 했으니 여기서 수동만)
        if is_login_like_page(driver) or (BASE_URL not in driver.current_url):
            wait_for_user_login(driver, timeout=180)

        print(f"현재 페이지: {driver.current_url}")

        print(f"\n날짜 설정 중: {start_date} ~ {end_date}")

        # 개선된 날짜 입력 (결과 반환)
        date_result = set_time_range_robust(driver, start_date, end_date, timeout=25)

        # 날짜 입력 결과 출력
        dates_ok = print_date_summary(date_result)

        if not dates_ok:
            print("⚠️ 경고: 날짜가 정확히 입력되지 않았습니다!")
            print("계속 진행하시겠습니까? (Enter=계속 / Ctrl+C=중단)")
            try:
                input()
            except KeyboardInterrupt:
                print("사용자에 의해 중단되었습니다.")
                return None

        print("\nSubmit 버튼 클릭 중...")
        click_submit(driver)

        print("데이터 로딩 대기 중...")
        wait_for_table(driver, timeout=30)
        time.sleep(2)

        print("\n테이블 데이터 추출 중...")
        df_list = pd.read_html(driver.page_source)

        if df_list:
            df = max(df_list, key=lambda x: len(x))
            print(f"✅ 데이터 추출 성공: {len(df)}개 행, {len(df.columns)}개 열")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"strategy_data_{timestamp}.xlsx"
            df.to_excel(filename, index=False)
            print(f"📁 파일 저장 완료: {filename}")

            # 최종 요약 출력
            print("\n" + "=" * 60)
            print("📊 최종 결과")
            print("=" * 60)
            print(f"  요청 기간: {start_date} ~ {end_date}")
            print(f"  실제 기간: {date_result['start_actual']} ~ {date_result['end_actual']}")
            print(f"  수집 행수: {len(df)}개")
            print(f"  저장 파일: {filename}")
            print("=" * 60)

            return df

        print("테이블 파싱 결과가 비어 있습니다.")
        screenshot_name = f"debug_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        driver.save_screenshot(screenshot_name)
        print(f"📸 스크린샷 저장: {screenshot_name}")
        return None

    except Exception as e:
        print(f"오류 발생: {e}")
        screenshot_name = f"error_screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        try:
            driver.save_screenshot(screenshot_name)
            print(f"📸 에러 스크린샷 저장: {screenshot_name}")
        except Exception:
            pass
        return None

    finally:
        print("\n브라우저를 닫으려면 Enter를 누르세요...")
        try:
            input()
        except Exception:
            pass
        driver.quit()


if __name__ == "__main__":
    print("=" * 60)
    print("전략 데이터 스크래퍼 v2.3 (날짜 검증 기능 추가)")
    print("=" * 60)

    print("\n옵션:")
    print("1. 오늘")
    print("2. 최근 7일")
    print("3. 날짜 직접 입력")

    choice = input("\n선택 (1-3): ").strip()
    today = datetime.now()

    if choice == "1":
        start_date = end_date = today.strftime("%Y-%m-%d")
    elif choice == "2":
        start_date = (today - timedelta(days=7)).strftime("%Y-%m-%d")
        end_date = today.strftime("%Y-%m-%d")
    else:
        start_date = input("시작 날짜 (YYYY-MM-DD): ").strip()
        end_date = input("종료 날짜 (YYYY-MM-DD): ").strip()

    result = scrape_strategy_data(start_date, end_date, headless=False)

    if result is not None:
        print("\n=== 결과 요약 ===")
        print(f"총 {len(result)}개 데이터 수집 완료")
        print("\n=== 데이터 미리보기 ===")
        print(result.head())
    else:
        print("\n데이터 수집 실패")
