import os
from dotenv import load_dotenv, find_dotenv
import json
import time
from functools import wraps
import logging
import requests
import asyncio
from playwright.async_api import async_playwright
import traceback
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "app.log")),
        logging.StreamHandler()
    ]
)
LOGGER = logging.getLogger()

class CentralBusinessConfiguration:
    def __init__(self) -> None:
        # Load environment variables from .env file
        load_dotenv(find_dotenv())
        self.LOGIN_URL = os.getenv("LOGIN_URL")
        self.EMAIL = os.getenv("EMAIL")
        self.PASSWORD = os.getenv("PASSWORD")
        self.DIR = os.path.dirname(os.path.abspath(__file__))
        # print(self.DIR)
    
    @staticmethod
    def ExceptionHandelling(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                LOGGER.info(f"⚠️EXCEPTION IN {func.__name__}: {e}")
                traceback.print_exc()
                return None
        return wrapper
    
    @ExceptionHandelling
    async def login(self):
        if not all([self.LOGIN_URL, self.EMAIL, self.PASSWORD]):
            raise ValueError("MISSING ENV VARIABLES")
        start_time = time.time()
        LOGGER.info("LOGGING IN...")
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-extensions",
                ],
            )
            context = await browser.new_context()
            page = await context.new_page()
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,css,woff,woff2,ttf}",
                lambda route: route.abort()
            )
            xsrf_token = None
            # ✅ Capture XSRF from REQUEST headers (outgoing)
            async def handle_request(request):
                nonlocal xsrf_token
                if request.method == "POST":
                    for k, v in request.headers.items():
                        if "xsrf" in k.lower():
                            xsrf_token = v
                            LOGGER.info(f"✅ XSRF FROM REQUEST: {v}")
                            page.off("request", handle_request)
            # ✅ Capture SAP-XSRF from RESPONSE headers (incoming) — from your screenshot
            async def handle_response(response):
                nonlocal xsrf_token
                for k, v in response.headers.items():
                    if "sap-xsrf" in k.lower() or "xsrf" in k.lower():
                        xsrf_token = v
                        LOGGER.info(f"✅ XSRF FROM RESPONSE: {v}")
                        page.remove_listener("response", handle_response)

            page.on("request", handle_request)
            page.on("response", handle_response)  # ✅ added this
            try:
                await page.goto(self.LOGIN_URL, wait_until="domcontentloaded")
                await page.fill('input[placeholder="E-Mail"]', self.EMAIL)
                await page.fill('input[placeholder="Password"]', self.PASSWORD)
                await page.press('input[placeholder="Password"]', "Enter")
                await page.wait_for_load_state("networkidle")

                # Get all cookies from browser context
                raw_cookies = await context.cookies()
                # Keep required CBC session cookies
                cookies = {
                    c["name"]: c["value"].strip('"')
                    for c in raw_cookies
                    if c["name"] in [
                        "BAF-STICK-SESSIONID",
                        "JSESSIONID",
                        "sap-cbc-sidx"   # 🔥 IMPORTANT
                    ]
                }
                LOGGER.info("COOKIES")
                for k, v in cookies.items():
                    LOGGER.info(f"  {k}: {v}")

                LOGGER.info(f"LOGIN DONE in {time.time() - start_time:.2f}s")

                return {
                    "COOKIES": cookies,
                    "XSRF-TOKEN": xsrf_token
                }
            except Exception as e:
                LOGGER.error(f"LOGIN FAILED: {e}", exc_info=True)
                return None

    @ExceptionHandelling
    def close(self):
        self.driver.quit()
        LOGGER.info("DRIVER CLOSED")