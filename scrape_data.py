from selenium import webdriver
import time

m_Options = webdriver.ChromeOptions()
# m_Options.add_argument("--headless")
# m_Options.add_argument("--no-sandbox")
m_Options.add_argument(
    "--user-data-dir=C:\\Users\\Sonu\\AppData\\Local\\Google\\Chrome\\User Data")
m_Options.add_argument("--profile-directory=Profile 1")
m_Options.add_argument("--disable-extensions")
driver = webdriver.Chrome(options=m_Options)

driver.get('http://mail.google.com')
first_mail = driver.find_element(by="class name", value="bog")

print("opening first mail...")
first_mail.click()

while (True):
    try:
        heading = driver.find_element(by="class name", value="hP").text
        sender_name = driver.find_element(by="class name", value="gD").text
        sender_email = driver.find_element(by="class name", value="go").text
        div_element = driver.find_element(
            by="class name", value="a3s")
        content = div_element.text

        # Moving onto next mail
        next_button = driver.find_element(
            by="xpath", value="//div[@aria-label='Older']")
        if (len(heading) == 0):
            print(f"Mail Not loaded. Retrying...", end='')
            time.sleep(0.5)
            print('\b \b'*100, end='')
            continue

        print(f"{heading=}")
        print(f"{sender_name=}")
        print(f"{sender_email=}")
        print(f"{div_element=}")
        print(f"{content=}")
        print(f"{next_button=}")
        print("Clicking next button")
        # input("Press enter to move to next email")
        next_button.click()
    except:
        print(f"Next button was not found. Retrying...", end='')
        time.sleep(0.5)
        print('\b \b'*100, end='')
        continue
    retry = 100

print("Done")
input("Press Enter to Quit")
driver.quit()
