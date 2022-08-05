import json
from selenium.common.exceptions import UnexpectedAlertPresentException
from utils import *


def download(driver, link):
    driver.get(link)
    download_btn = driver.find_element(By.CSS_SELECTOR, "button.btn-gd-primary.btn-round.save")
    download_btn.click()


def main():
    driver = webdriver.Chrome(options=init_driver_options())
    login(driver)

    f_l = open("file_list.txt", "r")
    file_list = json.load(f_l)

    for rec in file_list:
        try:
            download(driver, rec)
        except UnexpectedAlertPresentException:
            pass

    time.sleep(5)
    driver.quit()


if __name__ == '__main__':
    main()