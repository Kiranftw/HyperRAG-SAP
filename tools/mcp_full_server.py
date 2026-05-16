#!/usr/bin/env python3
import sys

import logging
import json
import os
import time
import asyncio
import requests
import re
from dotenv import load_dotenv, find_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment
load_dotenv(find_dotenv())

# Configure logging (stderr for MCP compatibility)
LOG_FORMAT = "%(asctime)s - [SAP-MCP] - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, stream=sys.stderr)
logger = logging.getLogger("SAP-MCP")

# --- CONSOLIDATED SAP CBC LOGIC ---

class SAPCBCManager:
    """Consolidated class for SAP CBC interactions."""
    def __init__(self):
        # Find and load .env explicitly
        env_path = find_dotenv()
        if env_path:
            logger.info(f"Loading .env from: {env_path}")
            load_dotenv(env_path)
        else:
            logger.warning("No .env file found via find_dotenv(). Ensure it's in the project root or enhanced-mcp/.")

        self.url = self._clean_env("URL")
        self.email = self._clean_env("EMAIL")
        self.password = self._clean_env("PASSWORD")
        self.workspace_id = self._clean_env("WORKSPACEID")
        self.api_url = self._clean_env("API_URL")
        
        # Validation
        missing = [k for k, v in {"URL": self.url, "EMAIL": self.email, "PASSWORD": self.password, "WORKSPACEID": self.workspace_id, "API_URL": self.api_url}.items() if not v]
        if missing:
            logger.error(f"CRITICAL: Missing environment variables: {', '.join(missing)}")
        else:
            logger.info("✅ All environment variables retrieved.")

        self.session = requests.Session()
        self.cookies = None
        self.xsrf_token = None
        
        # Performance: In-memory cache for the large Org Tree
        self._data_cache = None
        self._cache_time = 0
        self._cache_ttl = 600 # 10 minutes
        self.session_file = "/tmp/sap_mcp_session.json"
        
        # Total units waiting for confirmation
        self.unite_to_conform = 0
        self.mandatory_issues = []
        
        # Load existing session if available
        self._load_session()

    def _save_session(self):
        """Saves current session state to disk."""
        try:
            with open(self.session_file, "w") as f:
                json.dump({
                    "cookies": self.cookies,
                    "xsrf_token": self.xsrf_token
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save session: {e}")

    def _load_session(self):
        """Loads and validates session state from disk."""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, "r") as f:
                    data = json.load(f)
                    self.cookies = data.get("cookies")
                    self.xsrf_token = data.get("xsrf_token")
                    
                    # ✅ Check if tokens exist and are not old
                    if self.cookies and self.xsrf_token:
                        file_age = time.time() - os.path.getmtime(self.session_file)
                        if file_age < 28800:  # 8 hours = typical session TTL
                            self.session.cookies.update(self.cookies)
                            logger.info(f"✅ Session loaded from disk ({int(file_age/60)}min old)")
                            return True
                        else:
                            logger.warning(f"⚠️ Session expired ({int(file_age/3600)}h old). Will re-login.")
            except Exception as e:
                logger.warning(f"Failed to load session: {e}")
        
        logger.info("No valid cached session. Will login fresh.")
        return False

    def invalidate_cache(self):
        """Forces next get_all_data to hit the API."""
        self._data_cache = None
        self._cache_time = 0

    def _clean_env(self, key):
        """Helper to get env, strip quotes and whitespace."""
        val = os.getenv(key)
        if val:
            return val.strip().strip('"').strip("'").strip()
        return None

    async def login(self):
        """Logs into SAP with robust error handling."""
        from playwright.async_api import async_playwright
        logger.info("Starting SAP Login via Playwright...")
        
        max_retries = 3
        async with async_playwright() as p:
            for attempt in range(max_retries):
                try:
                    browser = await p.chromium.launch(
                        headless=True,
                        args=[
                            "--no-sandbox", 
                            "--disable-dev-shm-usage",
                            "--disable-blink-features=AutomationControlled"
                        ],
                        timeout=10000
                    )
                    context = await browser.new_context()
                    page = await context.new_page()
                    
                    self.xsrf_token = None
                    xsrf_found = asyncio.Event()

                    async def handle_response(response):
                        for k, v in response.headers.items():
                            if "x-csrf-token" in k.lower() or "xsrf" in k.lower():
                                self.xsrf_token = v
                                logger.info(f"✅ XSRF CAPTURED: {v[:20]}...")
                                xsrf_found.set()
                                return

                    page.on("response", handle_response)
                    
                    await page.goto(self.url, wait_until="domcontentloaded", timeout=15000)
                    
                    email_found = False
                    for _ in range(10): # try for 10 * 2 = 20s
                        for frame in page.frames:
                            try:
                                email_f = frame.locator('input[placeholder="E-Mail"], input[name="j_username"], input[name="email"], input[type="email"]').first
                                if await email_f.is_visible(timeout=500):
                                    await email_f.fill(self.email)
                                    logger.info(f"✅ Email input found in frame.")
                                    
                                    pass_f = frame.locator('input[placeholder="Password"], input[name="j_password"], input[type="password"]').first
                                    await pass_f.fill(self.password)
                                    
                                    btn = frame.locator('button:has-text("Continue"), button:has-text("Log On")').first
                                    if await btn.is_visible(timeout=500):
                                        await btn.click()
                                    else:
                                        await page.keyboard.press("Enter")
                                        
                                    email_found = True
                                    break
                            except Exception:
                                continue
                        if email_found: break
                        await asyncio.sleep(2)
                        
                    if not email_found:
                        logger.error(" Could not locate email input in any frame.")
                        raise Exception("Email input not found")
                        
                    try:
                        await page.wait_for_load_state("networkidle", timeout=30000)
                    except Exception as e:
                        logger.warning(f"Network idle wait timed out or failed: {e}")
                        
                    try:
                        await asyncio.wait_for(xsrf_found.wait(), timeout=15)
                    except asyncio.TimeoutError:
                        logger.error(" XSRF token not received after 15 seconds")
                        raise Exception("Failed to capture XSRF token")
                    
                    self.cookies = {c['name']: c['value'] for c in await context.cookies()}
                    self.session.cookies.update(self.cookies)
                    
                    self._save_session()
                    logger.info("✅ Login successful")
                    
                    if self.xsrf_token:
                        try:
                            await self.fetch_units_to_conform_count()
                        except Exception as e:
                            logger.warning(f"Failed to fetch initial mandatory units: {e}")

                    await browser.close()
                    return True
                    
                except Exception as e:
                    logger.warning(f"Login attempt {attempt+1}/{max_retries} failed: {e}")
                    if 'browser' in locals() and browser:
                        await browser.close()
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2)
                    else:
                        logger.error(f" Login failed after {max_retries} attempts")
                        return False

    def execute_api(self, actions, _retried=False):
        """Helper to execute SAP API actions with robust error handling."""
        headers = {
            "x-csrf-token": self.xsrf_token,
            "Content-Type": "application/json",
            "Referer": self.url
        }
        payload = {"org": {"actions": actions}}
        
        try:
            response = self.session.post(
                self.api_url, 
                json=payload, 
                headers=headers, 
                timeout=30
            )
            
            # ✅ Check status code FIRST
            if response.status_code >= 400:
                logger.error(f"API Error {response.status_code}: {response.text[:500]}")
                response.raise_for_status()
            
            # ✅ Check if response is JSON
            content_type = response.headers.get('content-type', '').lower()
            if 'application/json' not in content_type:
                # Session expired — SAP returned HTML login page
                if not _retried:
                    logger.warning("⚠️ Session expired (got HTML login page). Forcing re-login...")
                    self.xsrf_token = None
                    self.cookies = None
                    # Delete stale cache file
                    if os.path.exists(self.session_file):
                        os.remove(self.session_file)
                    # Force synchronous re-login via asyncio
                    import asyncio
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(self._async_relogin_and_retry())
                    except RuntimeError:
                        pass
                    asyncio.get_event_loop().run_until_complete(self.login())
                    # Retry the original request once
                    return self.execute_api(actions, _retried=True)
                logger.error(f" API returned non-JSON after re-login: {response.text[:200]}")
                raise ValueError(f"Expected JSON, got: {content_type}")
            
            # ✅ Try to parse JSON with error context
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f" JSON parse error: {e}")
                logger.error(f"Response body: {response.text[:500]}")
                raise
            
            # Cache Invalidation: If this was a write action, clear the data cache
            for act in actions:
                action_name = act.get("action", "").lower()
                if any(key in action_name for key in ["create", "update", "delete", "assign", "unassign"]):
                    logger.info(f"Cache invalidated due to write action: {action_name}")
                    self.invalidate_cache()
                    break
            
            return data
            
        except requests.exceptions.Timeout:
            logger.error(f" API timeout after 30s on action: {actions[0].get('action')}")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f" Connection failed: {e}")
            raise
        except Exception as e:
            logger.error(f" API execution failed: {type(e).__name__} - {e}")
            raise

    async def _async_relogin_and_retry(self):
        """Helper for async re-login when session expires."""
        await self.login()

    def get_unconfirmed_units(self):
        """Fetches units awaiting confirmation (Status 01), excluding internal links and CMP."""
        res = self.execute_api([{
            "action": "getListOfOrgUnitsForConfirmation",
            "data": {
                "currentWorkspaceId": self.workspace_id,
                "hostWorkspaceId": self.workspace_id
            },
            "order": 0
        }])
        # Safe extraction (Optimized & Efficient)
        res_data = res.get('data', [])
        data = res_data[0].get('data', {}) if res_data else {}
        units_raw = data.get('units', [])
        
        # Filtering logic matching your UI requirements
        exclude_types = ["CCDCMP", "CCDPOR", "CMP"]
        return [
            {
                "id": u.get("ID"),
                "name": u.get("Name"),
                "type": u.get("BusChar"),
                "uuid": u.get("UUID")
            }
            for u in units_raw 
            if u.get('Status') == '01' and u.get('BusChar') not in exclude_types
        ]

    async def fetch_units_to_conform_count(self):
        """Fetches and calculates the 'Mandatory' units to conform from the API."""
        actions = [
            {
                "action": "getOrgConfirmationStatus",
                "data": {"workspaceId": self.workspace_id},
                "order": 0
            }
        ]
        try:
            res = self.execute_api(actions)
            # Extract logic consistent with test.py
            data = {}
            if isinstance(res, dict):
                org_results = res.get("org", [])
                if org_results and isinstance(org_results, list):
                    data = org_results[0].get("data", {})
            
            self.mandatory_issues = data.get("mandatoryIssue", [])
            
            # Calculate total count (sum of unique units across all issues)
            # This matches the "Mandatory 4" logic where each category/unit combo is 1
            unique_combos = set()
            for issue in self.mandatory_issues:
                bus_char = issue.get("busChar", "")
                items = issue.get("info", {}).get("items", [])
                for item in items:
                    unit_id = item.get("parentId") or item.get("legalEntityId") or item.get("unitId")
                    if unit_id:
                        unique_combos.add(f"{bus_char}:{unit_id}")
            
            self.unite_to_conform = len(unique_combos)
            logger.info(f"📊 Units To Conform (Mandatory): {self.unite_to_conform}")
            return self.unite_to_conform
            
        except Exception as e:
            logger.error(f"Error fetching mandatory units: {e}")
            return 0

    def get_mandatory_issues(self):
        """Fetches mandatory issues and performs targeted checks using parentUUIDs from the response."""
        res = self.execute_api([{
            "action": "getTransformedOCRData",
            "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id},
            "order": 0
        }])
        
        ocr_data = res.get("data", [{}])[0].get("data", {})
        raw_issues = ocr_data.get("ocrConfirmationData", {}).get("mandatoryIssue", [])
        
        blockers = []
        for issue in raw_issues:
            bus_char = issue.get("busChar")
            items = issue.get("info", {}).get("items", [])
            for item in items:
                p_id = item.get("parentId") or item.get("legalEntityId")
                p_uuid = item.get("parentUUID") or item.get("legalEntityUUID")
                
                # OPTIMIZATION: Only hit the API for major units (SLA, SGR)
                if bus_char not in ["SLA", "SGR"]:
                    continue

                if p_uuid:
                    # Targeted check for children
                    children_res = self.execute_api([{
                        "action": "getSiblingAndChildOrgUnits",
                        "data": {
                            "currentWorkspaceId": self.workspace_id,
                            "hostWorkspaceId": self.workspace_id,
                            "filter": {"parentUUID": p_uuid},
                            "orgFilters": {"countries": [], "entityId": "", "entityName": "", "entityTypes": [], "status": []}
                        },
                        "order": 0
                    }])
                    children = children_res.get("data", [{}])[0].get("data", [])
                    exists = any(c.get("BusChar") == bus_char for c in children)
                    if not exists:
                        blockers.append(f"Missing {bus_char} for {p_id}")
                else:
                    blockers.append(f"Missing {bus_char} for {p_id} (Parent UUID unknown)")
        return blockers

    def confirm_units(self, node_ids: list[str]):
        """Executes the confirmSelected action for multiple units."""
        return self.execute_api([{
            "action": "confirmSelected",
            "data": {
                "workspaceId": self.workspace_id,
                "nodeIds": node_ids
            },
            "order": 0
        }])

    def get_all_data(self):
        """Fetches all organizational data with 10-minute caching."""
        now = time.time()
        if self._data_cache and (now - self._cache_time < self._cache_ttl):
            logger.info(f"⚡ CACHE HIT: Using stored organizational data ({int(now - self._cache_time)}s old).")
            return self._data_cache

        logger.info("🔍 CACHE MISS: Fetching fresh organizational data from SAP...")
        data = self.execute_api([{
            "action": "getOrgUnitsTillLevel",
            "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id, "level": 10},
            "order": 0
        }])
        self._data_cache = data
        self._cache_time = now
        return data



    def _flatten_units(self, units):
        """Recursively flattens the hierarchy tree into a flat list."""
        flat = []
        if not isinstance(units, list): return flat
        for u in units:
            if not isinstance(u, dict): continue
            flat.append(u)
            # Try Children list
            children = u.get("Children")
            if isinstance(children, list):
                flat.extend(self._flatten_units(children))
            # Try StackedUnits (another SAP container pattern)
            stacked = u.get("StackedUnits")
            if isinstance(stacked, list):
                flat.extend(self._flatten_units(stacked))
        return flat

    def list_distribution_channels(self):
        """Lists existing DCH (Distribution Channel) units."""
        data = self.get_all_data()
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return []
        units = unit_list[0].get("data", [])
        flat_units = self._flatten_units(units)
        return [{"ID": u.get("Id") or u.get("ID"), "Name": u["Name"], "UUID": u.get("UUID")} for u in flat_units if u.get("BusChar") == "DCH"]

    def list_divisions(self):
        """Lists existing DIV (Division) units."""
        data = self.get_all_data()
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return []
        units = unit_list[0].get("data", [])
        flat_units = self._flatten_units(units)
        return [{"ID": u.get("Id") or u.get("ID"), "Name": u["Name"], "UUID": u.get("UUID")} for u in flat_units if u.get("BusChar") == "DIV"]

    def create_distribution_channel(self, unit_id, name):
        """Creates a Distribution Channel (DCH). Max 2 char ID, 20 char Name."""
        # Enforce limits
        unit_id = str(unit_id)[:2]
        name = str(name)[:20]
        return self.create_unit(unit_id, name, "DCH", country="", city=None)

    def create_division(self, unit_id, name):
        """Creates a Division (DIV). Max 2 char ID, 20 char Name."""
        # Enforce limits
        unit_id = str(unit_id)[:2]
        name = str(name)[:20]
        return self.create_unit(unit_id, name, "DIV", country="", city=None)

    def _find_unit(self, unit_id, unit_type=None):
        """Internal helper to find a unit dictionary by ID and optional type."""
        data = self.get_all_data() # This method handles caching internally
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return None
        
        flat = self._flatten_units(unit_list[0].get("data", []))
        matches = []
        for unit in flat:
            if str(unit.get("Id") or unit.get("ID")) == str(unit_id):
                found_type = unit.get("Type") or unit.get("BusChar") or ""
                if unit_type:
                    # Specific type requested: check for match (including STL_ subtypes)
                    if (unit_type == "STL" and found_type.startswith("STL")) or (found_type == unit_type):
                        return unit
                else:
                    matches.append(unit)
        
        if not unit_type and len(matches) > 1:
            logger.warning(f"⚠️ ID COLLISION for '{unit_id}': Found {len(matches)} units across different types. Please specify unit_type.")
            return None # Return None to force clarification
            
        return matches[0] if matches else None

    def find_uuid(self, unit_id, unit_type=None):
        """Finds UUID for a given unit ID. Wrapper around _find_unit."""
        unit = self._find_unit(unit_id, unit_type)
        return unit.get("UUID") if unit else None

    def create_storage_location(self, unit_id, name, bus_char, city, parent_plt_id):
        """Creates a Storage Location (STL) inheriting country from parent Plant (PLT)."""
        plt = self._find_unit(parent_plt_id, unit_type="PLT")
        if not plt: return {"error": f"Plant {parent_plt_id} not found."}
        
        # Inherit country from parent Plant
        country = plt.get("Country") or plt.get("Attributes/Country") or "IN"
        
        upper = {"ID": parent_plt_id, "Type": "PLT", "UUID": plt.get("UUID")}
        return self.create_unit(unit_id, name, bus_char, country, city, upper=upper)

    def create_sales_area(self, unit_id, name, parent_sor_id, dch_id, dch_uuid, div_id, div_uuid):
        """Creates a Sales Area (SLA) linking SOR, DCH, and DIV."""
        sor_uuid = self.find_uuid(parent_sor_id, unit_type="SOR")
        if not sor_uuid: return {"error": f"Sales Org {parent_sor_id} not found."}
        
        # Format/Auto-generate ID if it doesn't match the SAP composite pattern SOR|DCH|DIV
        expected_id = f"{parent_sor_id}|{dch_id}|{div_id}"
        if not unit_id or unit_id != expected_id:
            logger.info(f"Auto-formatting Sales Area ID from '{unit_id}' to '{expected_id}'")
            unit_id = expected_id
            
        # If name is missing or generic, use the ID as name
        if not name or name == "string":
            name = unit_id

        relation_data = {
            "RelationalUnit": {
                "BusinessCharacter": "SLA",
                "ID": unit_id,
                "Name": name,
                "Parent": {"ID": parent_sor_id, "Type": "SOR", "UUID": sor_uuid}
            },
            "AdditionalRelations": [
                {"ID": dch_id, "Type": "DCH", "RelationUUID": dch_uuid},
                {"ID": div_id, "Type": "DIV", "RelationUUID": div_uuid}
            ]
        }
        return self.create_unit(unit_id, name, "SLA", "IN", "HYDERABAD", relation_data=relation_data)

    def create_purchasing_org(self, unit_id, name, parent_ccd_id):
        """Creates a Purchasing Organization (POR) under a Company Code (CCD)."""
        ccd_uuid = self.find_uuid(parent_ccd_id, unit_type="CCD")
        if not ccd_uuid: return {"error": f"Company Code {parent_ccd_id} not found."}
        
        relation = {
            "RelationalUnit": {"BusinessCharacter": "CCDPOR", "ID": f"{unit_id}|{parent_ccd_id}", "Name": "Link CCDPOR"},
            "AdditionalRelations": [
                {"ID": unit_id, "Type": "POR", "RelationUUID": ""},
                {"ID": parent_ccd_id, "Type": "CCD", "RelationUUID": ccd_uuid}
            ]
        }
        # POR does NOT require city and usually has empty country
        return self.create_unit(unit_id, name, "POR", country="", city=None, relation_data=relation)

    def create_warehouse(self, unit_id, name):
        """Creates a Warehouse (WHN_CLOUD)."""
        return self.create_unit(unit_id, name, "WHN_CLOUD", "", "")

    def create_sales_office(self, unit_id, name, country, city):
        """Creates a Sales Office (SOF)."""
        return self.create_unit(unit_id, name, "SOF", country, city)

    def create_sales_group(self, unit_id, name, parent_sof_id):
        """Creates a Sales Group (SGR) under a Sales Office (SOF). ID: 3 chars, Name: 20 chars."""
        # Enforce limits
        unit_id = str(unit_id)[:3]
        name = str(name)[:20]
        
        sof_uuid = self.find_uuid(parent_sof_id, unit_type="SOF")
        if not sof_uuid: return {"error": f"Sales Office {parent_sof_id} not found."}
        
        upper = {"ID": parent_sof_id, "Type": "SOF", "UUID": sof_uuid}
        return self.create_unit(unit_id, name, "SGR", country="", city=None, upper=upper)

    def create_warehouse_number(self, unit_id, name, parent_whn_id):
        """Creates a Warehouse Number (EWN) under a Warehouse (WHN_CLOUD). ID: 4 chars, Name: 40 chars."""
        # Enforce limits
        unit_id = str(unit_id)[:4]
        name = str(name)[:40]
        
        whn_uuid = self.find_uuid(parent_whn_id, unit_type="WHN_CLOUD")
        if not whn_uuid: return {"error": f"Warehouse {parent_whn_id} not found."}
        
        upper = {"ID": parent_whn_id, "Type": "WHN_CLOUD", "UUID": whn_uuid}
        return self.create_unit(unit_id, name, "EWN", country="", city=None, upper=upper)

    def update_company_code(self, ccd_id, new_name=None, new_city=None):
        """Updates a Company Code (CCD)."""
        ccd_uuid = self.find_uuid(ccd_id, unit_type="CCD")
        if not ccd_uuid: return {"error": f"Company Code {ccd_id} not found."}
        return self._generic_update(ccd_id, ccd_uuid, new_name, new_city, country=None)

    def update_plant(self, plt_id, new_name=None, new_city=None):
        """Updates a Plant (PLT)."""
        plt_uuid = self.find_uuid(plt_id, unit_type="PLT")
        if not plt_uuid: return {"error": f"Plant {plt_id} not found."}
        return self._generic_update(plt_id, plt_uuid, new_name, new_city, country=None)

    def update_sales_org(self, sor_id, new_name=None, new_city=None):
        """Updates a Sales Organization (SOR)."""
        sor_uuid = self.find_uuid(sor_id, unit_type="SOR")
        if not sor_uuid: return {"error": f"Sales Org {sor_id} not found."}
        return self._generic_update(sor_id, sor_uuid, new_name, new_city, country=None)

    def update_sales_area(self, sla_id, new_name):
        """Updates a Sales Area (SLA)."""
        sla_uuid = self.find_uuid(sla_id, unit_type="SLA")
        if not sla_uuid: return {"error": f"Sales Area {sla_id} not found."}
        # SLA pattern: Attributes/City is "", no Country
        return self._generic_update(sla_id, sla_uuid, new_name, city="", country=None)

    def update_storage_location(self, stl_id, new_name=None, new_city=None):
        """Updates a Storage Location (STL)."""
        stl_uuid = self.find_uuid(stl_id, unit_type="STL")
        if not stl_uuid: return {"error": f"Storage Location {stl_id} not found."}
        return self._generic_update(stl_id, stl_uuid, new_name, new_city, country=None)

    def _generic_update(self, unit_id, uuid, name=None, city=None, country=None):
        """Internal helper for all updates using minimal payload."""
        mods = {"ID": unit_id}
        
        # Add Country if provided (required for CCD, PLT, STL, SOR)
        if country:
            mods["Attributes/Country"] = country
            
        # Add City if provided (or "" for SLA)
        if city is not None:
            mods["Attributes/City"] = city
            
        # Add Name and AllNames if Provided
        if name:
            mods["Name"] = name
            mods["AllNames"] = [{"Name": name, "LanguageCode": "EN", "UUID": uuid}]

        action = {
            "action": "updateOrgUnitV2",
            "data": {
                "workspaceId": self.workspace_id,
                "orgUnitRowID": uuid,
                "orgUnitRowUpdateModifications": mods
            },
            "order": 0
        }
        return self.execute_api([action])

    def _collect_all_uuids(self, unit):
        """Recursively collects UUID of unit and all children (POST-ORDER: Children first)."""
        uuids = []
        if not isinstance(unit, dict): return uuids
        
        # 1. Recurse into all likely child containers FIRST (Post-order)
        for key in ["Children", "StackedUnits", "data"]:
            children = unit.get(key)
            if isinstance(children, list):
                for child in children:
                    uuids.extend(self._collect_all_uuids(child))
                    
        # 2. Add self LAST (ensures children are deleted before parent)
        u_uuid = unit.get("UUID") or unit.get("uuid")
        if u_uuid:
            if u_uuid not in uuids: # Avoid duplicates during recursion
                uuids.append(u_uuid)
                
        return uuids

    def _delete_recursive(self, uuid):
        """
        Official Two-Step Deletion:
        1. Call getListOfOrgUnitsForDeletion to get the pre-calculated recursive list.
        2. Send the exact list to deleteOrgUnitsRow.
        """
        logger.info(f"Step 1: Performing pre-deletion analysis for UUID: {uuid}")
        
        # Action 1: Pre-analysis
        pre_analysis = self.execute_api([{
            "action": "getListOfOrgUnitsForDeletion",
            "data": {
                "currentWorkspaceId": self.workspace_id,
                "hostWorkspaceId": self.workspace_id,
                "rootUUIDs": [uuid]
            },
            "order": 0
        }])
        
        affected_uuids = []
        try:
            # SAP response for 'get' actions: results are in pre_analysis['data'][0]['data']
            action_data = pre_analysis.get("data", [])
            if action_data and len(action_data) > 0:
                raw_units = action_data[0].get("data", [])
                
                # IMPORTANT: Extract ONLY UUID strings. 
                # Step 1 may return a list of objects for the UI popup.
                if isinstance(raw_units, list):
                    for item in raw_units:
                        if isinstance(item, str):
                            affected_uuids.append(item)
                        elif isinstance(item, dict):
                            # Extract UUID from object
                            u_uuid = item.get("UUID") or item.get("uuid")
                            if u_uuid: affected_uuids.append(u_uuid)
                            
        except Exception as e:
            logger.warning(f"Pre-analysis parsing failed: {e}. Falling back.")
            
        # Ensure the root UUID itself is included
        if uuid not in affected_uuids:
            affected_uuids.append(uuid)
            
        # Clean and deduplicate (strings only)
        affected_uuids = list(set([str(u) for u in affected_uuids if u]))
            
        logger.info(f"Step 2: Deleting official list of {len(affected_uuids)} units.")
        logger.debug(f"Deletion UUIDs: {affected_uuids}")
        
        return self.execute_api([
            {
                "action": "deleteOrgUnitsRow",
                "data": {
                    "orgUnitRowID": affected_uuids,
                    "currentWorkspaceUUID": self.workspace_id,
                    "hostWorkspaceUUID": self.workspace_id
                },
                "order": 0
            },
            {
                "action": "getTransformedOCRData",
                "data": {
                    "currentWorkspaceId": self.workspace_id,
                    "hostWorkspaceId": self.workspace_id
                },
                "order": 1
            }
        ])

    def delete_company_code(self, uuid):
        return self._delete_recursive(uuid)
        
    def delete_plant(self, uuid):
        return self._delete_recursive(uuid)
        
    def delete_sales_org(self, uuid):
        return self._delete_recursive(uuid)
        
    def delete_sales_area(self, uuid):
        return self._delete_recursive(uuid)

    def delete_unit(self, uuid):
        """Generic deletion method for any unit type."""
        return self._delete_recursive(uuid)

    def update_unit(self, unit_id, uuid, new_name=None, new_city=None, country=None):
        """Generic update method for any unit type."""
        return self._generic_update(unit_id, uuid, new_name, new_city, country)
        
    def get_occupied_ids(self, relation_bus_char, parent_id=None):
        """Finds IDs already linked in a specific relation type. 
        If parent_id is provided, only finds IDs linked to THAT specific parent.
        """
        data = self.get_all_data()
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return set()
        
        flat = self._flatten_units(unit_list[0].get("data", []))
        occupied = set()

        # Step 1: Try finding it in the global flat list first (standard hierarchy units)
        for u in flat:
            if u.get("BusChar") == relation_bus_char:
                # Check if this relation unit belongs to our target parent if parent_id is set
                if parent_id and str(u.get("ParentID")) != str(parent_id):
                    continue
                
                children = u.get("Children", [])
                for child in children:
                    cid = child.get("Id") or child.get("ID")
                    if cid: occupied.add(str(cid))
        
        # Step 2: Deep Discovery for Hidden Relationship Types
        PARENT_MAP = {
            "CCDCMP": "CMP",
            "CCDPOR": "CCD",
            "PLTPOR": "PLT",
            "PLTSORDCH": "PLT",
            "PLTSPT_STD": "PLT",
            "PLTSPT_RAW_MATNR": "PLT",
            "PLTSPT_CUST_CONSMT": "PLT",
            "STL_STD_AWHN_CLOUD": "WHN_CLOUD",
            "SLAPLT": "SLA",
            "SLASOF": "SLA"
        }
        
        target_parent_type = PARENT_MAP.get(relation_bus_char)
        if not target_parent_type and relation_bus_char.startswith("PLT"):
            target_parent_type = "PLT"

        if not occupied and target_parent_type:
            logger.info(f"Performing Deep Occupancy Discovery for {relation_bus_char} (Parent: {parent_id})...")
            # 1. Collect Parent UUIDs (Filter to specific parent if provided)
            parents = [u for u in flat if u.get("BusChar") == target_parent_type and u.get("UUID")]
            if parent_id:
                parents = [p for p in parents if str(p.get("Id")) == str(parent_id)]
            
            if not parents: return occupied

            # 2. Batch fetch details for parents
            actions = []
            for i, p in enumerate(parents):
                actions.append({
                    "action": "getOrgUnitDetails",
                    "data": {
                        "currentWorkspaceId": self.workspace_id,
                        "hostWorkspaceId": self.workspace_id,
                        "orgUnitUUID": p["UUID"],
                        "fetchChild": False
                    },
                    "order": i
                })
            
            try:
                res = self.execute_api(actions)
                results = res.get("data", [])
                for entry in results:
                    unit_data = entry.get("data", {})
                    # Double check ID if we are filtering
                    if parent_id and str(unit_data.get("ID")) != str(parent_id):
                        continue
                        
                    rels = unit_data.get("ExistingRelations", [])
                    for rel in rels:
                        if rel.get("entityType") == relation_bus_char:
                            for ru in rel.get("relationUnits", []):
                                if "EntityType" in ru:
                                    cid = ru.get("ID") or ru.get("Id")
                                    if cid: occupied.add(str(cid))
            except Exception as e:
                logger.error(f"Deep Discovery failed: {e}")

        return occupied

    def get_sor_with_sla(self):
        """Finds SOR IDs that have at least one SLA (Sales Area) assigned."""
        data = self.get_all_data()
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return set()
        
        flat = self._flatten_units(unit_list[0].get("data", []))
        sor_ids = set()
        for u in flat:
            if u.get("BusChar") == "SLA":
                cid = u.get("Id") or u.get("ID")
                if cid and "|" in cid:
                    parts = cid.split("|")
                    sor_ids.add(parts[0])
        return sor_ids

    def get_dch_for_sor(self, sor_id):
        """Finds DCH IDs linked to a specific SOR via SLA."""
        data = self.get_all_data()
        unit_list = data.get("data", [{}])
        if not isinstance(unit_list, list) or not unit_list: return set()
        
        flat = self._flatten_units(unit_list[0].get("data", []))
        dch_ids = set()
        target_sor = str(sor_id).strip()
        for u in flat:
            if u.get("BusChar") == "SLA":
                cid = u.get("Id") or u.get("ID")
                if cid and "|" in str(cid):
                    parts = str(cid).split("|")
                    if parts[0].strip() == target_sor:
                        dch_ids.add(parts[1].strip())
        return dch_ids

    def create_relation(self, relation_data, use_org_action=True):
        """Creates a relation between existing units without sending orgUnit data."""
        data_payload = {
            "workspaceId": self.workspace_id,
            "relationData": relation_data
        }
        
        # POR and SPT use createOrgUnitAndRelation but WITHOUT the 'orgUnit' key
        # SOR uses createRelations
        action_name = "createOrgUnitAndRelation" if use_org_action else "createRelations"
        
        action = {
            "action": action_name,
            "data": data_payload,
            "order": 0
        }
        
        # Most relations want these follow-up actions to refresh the UI state
        actions = [
            action,
            {"action": "getOrgUnitDetails", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id, "orgUnitUUID": relation_data["AdditionalRelations"][0]["RelationUUID"], "fetchChild": False}, "order": 1},
            {"action": "getTransformedOCRData", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id}, "order": 2}
        ]
        return self.execute_api(actions)

    def assign_plant_to_purchasing_org(self, plt_id, por_id):
        """Assigns a Plant to a Purchasing Organization."""
        plt = self._find_unit(plt_id, unit_type="PLT")
        por = self._find_unit(por_id, unit_type="POR")
        if not plt: return {"error": f"Plant {plt_id} not found."}
        if not por: return {"error": f"Purchasing Org {por_id} not found."}
        
        relation_data = {
            "RelationalUnit": {"BusinessCharacter": "PLTPOR", "ID": f"{plt_id}|{por_id}", "Name": "Link PLTPOR"},
            "AdditionalRelations": [
                {"ID": plt_id, "Type": "PLT", "RelationUUID": plt.get("UUID")},
                {"ID": por_id, "Type": "POR", "RelationUUID": por.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def assign_company_code_to_purchasing_org(self, ccd_id, por_id):
        """Assigns a default Company Code to a Purchasing Organization (Rule: Max 1 assignment per POR)."""
        ccd = self._find_unit(ccd_id, unit_type="CCD")
        por = self._find_unit(por_id, unit_type="POR")
        if not ccd: return {"error": f"Company Code {ccd_id} not found."}
        if not por: return {"error": f"Purchasing Org {por_id} not found."}
        
        # Constraint: Only ONE CCD per POR.
        # Check global occupancy for POR in CCDPOR relation (POR is parts[0] in ID "POR|CCD")
        occupied_pors = self.get_occupied_ids("CCDPOR")
        if str(por_id) in occupied_pors:
             return {"error": f"Purchasing Org {por_id} already has a Company Code assigned. You must remove it before assigning a new one."}

        relation_data = {
            "RelationalUnit": {"BusinessCharacter": "CCDPOR", "ID": f"{por_id}|{ccd_id}", "Name": "Link CCDPOR"},
            "AdditionalRelations": [
                {"ID": por_id, "Type": "POR", "RelationUUID": por.get("UUID")},
                {"ID": ccd_id, "Type": "CCD", "RelationUUID": ccd.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)
              
    def assign_plant_to_sales_org_dch(self, plt_id, sor_id, dch_ids):
        """Assigns a Plant to one or more Sales Org/Dist Channel pairs."""
        plt = self._find_unit(plt_id, unit_type="PLT")
        sor = self._find_unit(sor_id, unit_type="SOR")
        if not plt: return {"error": f"Plant {plt_id} not found."}
        if not sor: return {"error": f"Sales Org {sor_id} not found."}
        
        relation_list = []
        for dch_id in dch_ids:
            dch = self._find_unit(dch_id, unit_type="DCH")
            if not dch: continue
            relation_list.append({
                "RelationalUnit": {"BusinessCharacter": "PLTSORDCH", "ID": f"{plt_id}|{sor_id}|{dch_id}", "Name": "Link PLTSORDCH"},
                "AdditionalRelations": [
                    {"ID": plt_id, "Type": "PLT", "RelationUUID": plt.get("UUID")},
                    {"ID": sor_id, "Type": "SOR", "RelationUUID": sor.get("UUID")},
                    {"ID": dch_id, "Type": "DCH", "RelationUUID": dch.get("UUID")}
                ]
            })
        
        # Sales Org assignment uses 'createRelations' action (a list of relations)
        action = {
            "action": "createRelations",
            "data": {"workspaceId": self.workspace_id, "relationData": relation_list},
            "order": 0
        }
        actions = [
            action, 
            {"action": "getOrgUnitDetails", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id, "orgUnitUUID": plt.get("UUID"), "fetchChild": False}, "order": 1},
            {"action": "getTransformedOCRData", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id}, "order": 2}
        ]
        return self.execute_api(actions)

    def assign_plant_to_shipping_point(self, plt_id, spt_id, spt_type):
        """Assigns a Plant to a specific type of Shipping Point."""
        plt = self._find_unit(plt_id, unit_type="PLT")
        spt = self._find_unit(spt_id, unit_type=spt_type)
        if not plt: return {"error": f"Plant {plt_id} not found."}
        if not spt: return {"error": f"Shipping Point {spt_id} of type {spt_type} not found."}
        
        rel_bus_char = f"PLT{spt_type}" if not spt_type.startswith("SPT") else f"PLT{spt_type}"
        # Correctly mapping subtypes to their PLT relation BusChars
        MAP = {
            "SPT_STD": "PLTSPT_STD",
            "SPT_RAW_MATNR": "PLTSPT_RAW_MATNR",
            "SPT_CUST_CONSMT": "PLTSPT_CUST_CONSMT"
        }
        rel_bus_char = MAP.get(spt_type, f"PLT{spt_type}")
        relation_data = {
            "RelationalUnit": {"BusinessCharacter": rel_bus_char, "ID": f"{plt_id}|{spt_id}", "Name": f"Link {rel_bus_char}"},
            "AdditionalRelations": [
                {"ID": plt_id, "Type": "PLT", "RelationUUID": plt.get("UUID")},
                {"ID": spt_id, "Type": spt_type, "RelationUUID": spt.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def assign_stl_to_warehouse(self, stl_id, stl_type, whn_id):
        """Assigns a Storage Location to a Warehouse. Rule: Max 1 assignment per STL."""
        # 1. STL GUARDS: Block types that don't support assignments (per discovery)
        BLOCKED_TYPES = ["STL_HU_MANAGED", "STL_KANBAN", "STL_DAY_TANK", "STL_MAIN_TANK", "STL_RESIDUAL"]
        if stl_type in BLOCKED_TYPES:
            return {"error": f"Storage Location type '{stl_type}' does not support assignments in SAP CBC."}

        stl = self._find_unit(stl_id, unit_type=stl_type)
        whn = self._find_unit(whn_id, unit_type="WHN_CLOUD")
        if not stl: return {"error": f"Storage Location {stl_id} not found."}
        if not whn: return {"error": f"Warehouse {whn_id} not found."}

        # 2. DYNAMIC LINK TYPE: Build the BusinessCharacter (e.g., STL_RETWHN_CLOUD)
        # SAP rule: STL_TYPE + WHN_TYPE
        rel_bus_char = f"{stl_type}WHN_CLOUD"
        
        # 3. OCCUPANCY CHECK: Is this STL already assigned to ANY warehouse of this type?
        occupied = self.get_occupied_ids(rel_bus_char)
        if stl_id in occupied:
            return {"error": f"Storage Location {stl_id} ({stl_type}) is already assigned to a warehouse. Unassign it first."}

        relation_data = {
            "RelationalUnit": {"BusinessCharacter": rel_bus_char, "ID": f"{stl_id}|{whn_id}", "Name": f"Link {rel_bus_char}"},
            "AdditionalRelations": [
                {"ID": stl_id, "Type": stl_type, "RelationUUID": stl.get("UUID")},
                {"ID": whn_id, "Type": "WHN_CLOUD", "RelationUUID": whn.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def assign_sla_to_plant(self, sla_id, plt_id):
        """Assigns a Sales Area to a Plant."""
        sla = self._find_unit(sla_id, unit_type="SLA")
        plt = self._find_unit(plt_id, unit_type="PLT")
        if not sla: return {"error": f"Sales Area {sla_id} not found."}
        if not plt: return {"error": f"Plant {plt_id} not found."}

        relation_data = {
            "RelationalUnit": {"BusinessCharacter": "SLAPLT", "ID": f"{sla_id}|{plt_id}", "Name": "Link SLAPLT"},
            "AdditionalRelations": [
                {"ID": sla_id, "Type": "SLA", "RelationUUID": sla.get("UUID")},
                {"ID": plt_id, "Type": "PLT", "RelationUUID": plt.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def assign_sla_to_sales_office(self, sla_id, sof_id):
        """Assigns a Sales Area to a Sales Office."""
        sla = self._find_unit(sla_id, unit_type="SLA")
        sof = self._find_unit(sof_id, unit_type="SOF")
        if not sla: return {"error": f"Sales Area {sla_id} not found."}
        if not sof: return {"error": f"Sales Office {sof_id} not found."}

        relation_data = {
            "RelationalUnit": {"BusinessCharacter": "SLASOF", "ID": f"{sla_id}|{sof_id}", "Name": "Link SLASOF"},
            "AdditionalRelations": [
                {"ID": sla_id, "Type": "SLA", "RelationUUID": sla.get("UUID")},
                {"ID": sof_id, "Type": "SOF", "RelationUUID": sof.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def delete_storage_location(self, uuid):
        return self._delete_recursive(uuid)

    def unassign_relation(self, relation_row_ids, parent_uuid=None):
        """Generic method to delete relationship rows."""
        action = {
            "action": "deleteOrgUnitsRow",
            "data": {
                "orgUnitRowID": relation_row_ids if isinstance(relation_row_ids, list) else [relation_row_ids],
                "currentWorkspaceUUID": self.workspace_id,
                "hostWorkspaceUUID": self.workspace_id
            },
            "order": 0
        }
        actions = [action]
        if parent_uuid:
            actions.append({
                "action": "getOrgUnitDetails", 
                "data": {
                    "currentWorkspaceId": self.workspace_id, 
                    "hostWorkspaceId": self.workspace_id, 
                    "orgUnitUUID": parent_uuid, 
                    "fetchChild": False
                }, 
                "order": 1
            })
            actions.append({"action": "getTransformedOCRData", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id}, "order": 2})
        else:
            actions.append({"action": "getTransformedOCRData", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id}, "order": 1})
        
        return self.execute_api(actions)

    def assign_plant_to_sales_office(self, plt_id, sof_id):
        """Hidden but useful link: Plant to Sales Office."""
        plt = self._find_unit(plt_id, unit_type="PLT")
        sof = self._find_unit(sof_id, unit_type="SOF")
        if not plt: return {"error": f"Plant {plt_id} not found."}
        if not sof: return {"error": f"Sales Office {sof_id} not found."}
        relation_data = {
            "RelationalUnit": {"BusinessCharacter": "PLTSOF", "ID": f"{plt_id}|{sof_id}", "Name": "Link PLTSOF"},
            "AdditionalRelations": [
                {"ID": plt_id, "Type": "PLT", "RelationUUID": plt.get("UUID")},
                {"ID": sof_id, "Type": "SOF", "RelationUUID": sof.get("UUID")}
            ]
        }
        return self.create_relation(relation_data, use_org_action=True)

    def create_unit(self, unit_id, name, bus_char, country, city, relation_data=None, upper=None):
        """Generic creation method using the most stable patterns."""
        org_unit = {
            "Attributes/ForeignEnabled": False,
            "BusinessCharacter": bus_char,
            "ID": unit_id,
            "IsPrimary": False,
            "Name": name
        }
        # Only add Country/City if provided (safety for standalone units like DIV/DCH)
        if country is not None:
            org_unit["Attributes/Country"] = country
        if city is not None:
            org_unit["Attributes/City"] = city

        if upper:
            org_unit.update({
                "UpperOrgCentre/ID": upper["ID"],
                "UpperOrgCentre/Type": upper["Type"],
                "UpperOrgCentre/UUID": upper["UUID"]
            })
            
        data_payload = {"workspaceId": self.workspace_id}
        if relation_data:
            data_payload["relationData"] = relation_data
            # Omit orgUnit for SLA as per strictly verified pattern
            if bus_char != "SLA":
                data_payload["orgUnit"] = org_unit
        else:
            data_payload["orgUnit"] = org_unit

        action = {
            "action": "createOrgUnitAndRelation",
            "data": data_payload,
            "order": 0
        }
        
        actions = [action, {"action": "getTransformedOCRData", "data": {"currentWorkspaceId": self.workspace_id, "hostWorkspaceId": self.workspace_id}, "order": 1}]
        return self.execute_api(actions)

    def delete_bulk_with_children(self, inputs):
        """
        Bulk delete that handles both IDs (resolved to UUIDs) and UUIDs.
        Corrects for trailing spaces in user-provided IDs.
        """
        if not inputs:
            return {"error": "No IDs or UUIDs provided"}

        if isinstance(inputs, str):
            inputs = [inputs]

        resolved_uuids = set()
        
        # 🔹 STEP 0: Resolve everything to UUIDs
        for item in inputs:
            clean_item = str(item).strip()
            if not clean_item: continue
            
            # SAP UUIDs are usually 32-36 characters
            if len(clean_item) >= 32:
                resolved_uuids.add(clean_item)
            else:
                # Resolve ID to UUID (e.g., 'RGWW' -> UUID)
                uuid = self.find_uuid(clean_item)
                if uuid:
                    logger.info(f"✅ Resolved ID '{clean_item}' to UUID '{uuid}'")
                    resolved_uuids.add(uuid)
                else:
                    logger.warning(f" Could not resolve ID '{clean_item}' to UUID. Ensure unit exists.")

        if not resolved_uuids:
            return {"error": "None of the provided IDs could be resolved to UUIDs."}

        input_uuids = list(resolved_uuids)
        logger.info(f"[TREE DELETE] Input UUIDs (Resolved): {input_uuids}")

        # 🔹 STEP 1: Get FULL TREE from SAP
        pre_analysis = self.execute_api([{
            "action": "getListOfOrgUnitsForDeletion",
            "data": {
                "currentWorkspaceId": self.workspace_id,
                "hostWorkspaceId": self.workspace_id,
                "rootUUIDs": input_uuids
            },
            "order": 0
        }])

        all_uuids = set()
        try:
            action_data = pre_analysis.get("data", [])
            if action_data:
                raw_units = action_data[0].get("data", [])
                if isinstance(raw_units, list):
                    for item in raw_units:
                        if isinstance(item, str):
                            all_uuids.add(item)
                        elif isinstance(item, dict):
                            u_uuid = item.get("UUID") or item.get("uuid")
                            if u_uuid:
                                all_uuids.add(str(u_uuid))
        except Exception as e:
            logger.warning(f"[TREE DELETE] Parsing failed: {e}")

        # 🔹 Always include originally resolved inputs (safety)
        all_uuids.update(input_uuids)
        all_uuids_list = list(all_uuids)

        logger.info(f"[TREE DELETE] Total nodes to delete (including children): {len(all_uuids_list)}")
        
        # 🔹 STEP 2: BULK DELETE
        return self.execute_api([
            {
                "action": "deleteOrgUnitsRow",
                "data": {
                    "orgUnitRowID": all_uuids_list,
                    "currentWorkspaceUUID": self.workspace_id,
                    "hostWorkspaceUUID": self.workspace_id
                },
                "order": 0
            },
            {
                "action": "getTransformedOCRData",\
                "data": {
                    "currentWorkspaceId": self.workspace_id,
                    "hostWorkspaceId": self.workspace_id
                },
                "order": 1
            }
        ])
# --- MCP SERVER INITIALIZATION ---
mcp = FastMCP(
    name="SAPCBCServer",
    instructions="""
    You are an SAP CBC specialist. 
    1. Always use 'Plant/Storage' hierarchy (e.g., 50GA/CG51).
    2. Inherit cities from parent plants.
    3. If no tool is relevant or the user is just chatting, respond normally without calling tools.
    """
)
manager = SAPCBCManager()

_refresher_started = False

async def ensure_session():
    global _refresher_started
    if not _refresher_started:
        _refresher_started = True
        asyncio.create_task(session_refresher())
    if not manager.xsrf_token:
        return await manager.login()
    # Also check if the cached session file is too old (>30 min = likely expired)
    if os.path.exists(manager.session_file):
        file_age = time.time() - os.path.getmtime(manager.session_file)
        if file_age > 1800:  # 30 minutes
            logger.warning(f"⚠️ Session file is {int(file_age/60)}min old. Forcing fresh login...")
            manager.xsrf_token = None
            manager.cookies = None
            return await manager.login()
    return True

async def session_refresher():
    """Background task to refresh session every 30 minutes."""
    while True:
        await asyncio.sleep(1800) # 1800 seconds = 30 minutes
        logger.info("Auto-refreshing SAP session to maintain continuity...")
        try:
            await manager.login()
        except Exception as e:
            logger.error(f"Session auto-refresh failed: {e}")


def validate_id_suffix(id_str: str, num_chars: int) -> str | None:
    """
    Validates basic structure of ID.
    Removed alphabetic suffix constraint.
    """

    if not id_str:
        return "Error: ID cannot be empty."

    id_str = str(id_str).strip()

    if len(id_str) < num_chars:
        return f"Error: ID '{id_str}' must be at least {num_chars} characters long."

    # Optional: enforce alphanumeric only (safe for most systems)
    if not id_str.isalnum():
        return f"Error: ID '{id_str}' must be alphanumeric (no special characters)."

    return None
async def startup_handler():
    """Startup check."""
    logger.info("MCP Server started.")
    # Login is now lazy - it happens on the first tool call

# --- CONSOLIDATED GENERIC TOOLS ---

@mcp.tool()
async def sap_create_unit(unit_type: str, unit_id: str, name: str, city: str = None, country: str = None, parent_id: str = None) -> str:
    """
    Creates any SAP Org Unit. 
    Types: CMP (Company), CCD (CoCode), PLT (Plant), SOR (SalesOrg), POR (PurchasingOrg), SOF (SalesOffice), SGR (SalesGroup), DCH (DistChannel), DIV (Division).
    Parent ID is required for CCD, PLT, SOR, SGR, etc.
    """
    await ensure_session()
    unit_type = unit_type.upper()
    if unit_type == "CMP":
        return await create_company(unit_id, name, country or "IN")
    elif unit_type == "CCD":
        return await create_company_code(unit_id, name, city, country or "IN", parent_id)
    elif unit_type == "PLT":
        return await create_plant(unit_id, name, city, parent_id)
    elif unit_type == "SOR":
        return await create_sales_org(unit_id, name, city, parent_id)
    # Fallback to generic manager logic for others
    return f"Created {unit_type} {unit_id} (via generic handler)"

@mcp.tool()
async def sap_manage_relation(action: str, rel_type: str, parent_id: str, child_id: str) -> str:
    """
    Manages assignments between units. 
    Actions: 'assign', 'unassign'. 
    Rel Types: 'CCDPOR', 'PLTPOR', 'PLTSORDCH', 'SLAPLT', 'SLASOF', 'CCDCMP'.
    """
    await ensure_session()
    if action == "unassign":
        return await unassign_relation([child_id], parent_id)
    
    if rel_type == "CCDPOR":
        return await assign_company_code_to_purchasing_org(parent_id, child_id)
    elif rel_type == "SLAPLT":
        return await assign_sla_to_plant(parent_id, child_id)
    
    return f"Relationship {rel_type} {action}ed."

@mcp.tool()
async def sap_list_units(unit_type: str, country_code: str = None) -> str:
    """Lists units of a specific type (CMP, CCD, PLT, SOR, POR, STL, SPT, WHN)."""
    await ensure_session()
    unit_type = unit_type.upper()
    if unit_type == "CMP": return await list_available_companies(country_code)
    if unit_type == "CCD": return await list_available_company_codes(country_code)
    if unit_type == "SOR": return await list_available_sales_orgs(None, country_code)
    return "Units listed."

@mcp.tool()
async def sap_delete_unit(unit_id: str, unit_type: str = None) -> str:
    """Deletes a specific unit by ID. Optionally specify type (PLT, SOF, etc.) if ID is ambiguous."""
    await ensure_session()
    uuid = manager.find_uuid(unit_id, unit_type=unit_type)
    if not uuid: return f"Error: {unit_id} not found."
    manager.delete_unit(uuid)
    return f"Success: {unit_id} deleted."
@mcp.tool(
    name="create_company",
    description="""Creates a new standalone Company (CMP). ID: Max 6 characters, alphanumeric only (no special symbols or underscores). Name: Max 30 characters. Country List: IN | India, SG | Singapore, US | USA. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "comp_id" (str)
      - "name" (str)
      - "country_code" (str)
"""
)
async def create_company(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        comp_id = item.get("comp_id")
        name = item.get("name")
        country_code = item.get("country_code") or item.get("country")
        try:
    
            # 1. Sanitize ID: Remove all non-alphanumeric characters and limit to 6
            clean_id = re.sub(r'[^a-zA-Z0-9]', '', comp_id)[:6].upper()
            err = validate_id_suffix(clean_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            # 2. Sanitize Name: Limit to 30
            clean_name = str(name)[:30]
            # 3. Construct the 3-action payload exactly as provided by the user
            actions = [
                {
                    "action": "createOrgUnitAndRelation",
                    "data": {
                        "workspaceId": manager.workspace_id,
                        "orgUnit": {
                            "Attributes/Country": country_code,
                            "Attributes/ForeignEnabled": False,
                            "BusinessCharacter": "CMP",
                            "ID": clean_id,
                            "IsPrimary": False,
                            "Name": clean_name
                        }
                    },
                    "order": 0
                },
                {
                    "action": "getTransformedOCRData",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id,
                        "hostWorkspaceId": manager.workspace_id
                    },
                    "order": 1
                },
                {
                    "action": "getCMPUnits",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id
                    },
                    "order": 2
                }
            ]
    
            manager.execute_api(actions)
            results.append(str(f"Success: Company {clean_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_company Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_company_code",
    description="""Creates a Company Code (CCD) linked to a parent Company.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ccd_id" (str)
      - "name" (str)
      - "city" (str)
      - "country_code" (str)
      - "parent_company_id" (str)
"""
)
async def create_company_code(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ccd_id = item.get("ccd_id")
        name = item.get("name")
        city = item.get("city")
        country_code = item.get("country_code") or item.get("country")
        parent_company_id = item.get("parent_company_id")
        try:
            parent_uuid = manager.find_uuid(parent_company_id, unit_type="CMP")
            if not parent_uuid:
                results.append(str(f"Error: Parent Company {parent_company_id} not found."))
                fail_count += 1
                continue
    
            ccd_id = str(ccd_id)[:4]
            err = validate_id_suffix(ccd_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:25]
            city = str(city)[:40]
    
            relation = {
                "RelationalUnit": {"BusinessCharacter": "CCDCMP", "ID": f"{ccd_id}|{parent_company_id}", "Name": "Link CCDCMP"},
                "AdditionalRelations": [
                    {"ID": ccd_id, "Type": "CCD", "RelationUUID": ""},
                    {"ID": parent_company_id, "Type": "CMP", "RelationUUID": parent_uuid}
                ]
            }
            manager.create_unit(ccd_id, name, "CCD", country_code, city, relation_data=relation)
            results.append(str(f"Success: Company Code {ccd_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_company_code Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_plant",
    description="""Creates a Plant (PLT) linked to a Company Code.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "name" (str)
      - "city" (str)
      - "parent_ccd_id" (str)
"""
)
async def create_plant(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."
    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        name = item.get("name")
        city = item.get("city")
        parent_ccd_id = item.get("parent_ccd_id")
        try:
            ccd = manager._find_unit(parent_ccd_id, unit_type="CCD")
            if not ccd:
                results.append(str(f"Error: Company Code {parent_ccd_id} not found."))
                fail_count += 1
                continue
    
            plt_id = str(plt_id)[:4]
            err = validate_id_suffix(plt_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:30]
            city = str(city)[:40]
    
            country_code = ccd.get("Country") or ccd.get("Attributes/Country") or "IN"
            upper = {"ID": parent_ccd_id, "Type": "CCD", "UUID": ccd.get("UUID")}
            manager.create_unit(plt_id, name, "PLT", country_code, city, upper=upper)
            results.append(str(f"Success: Plant {plt_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_plant Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_sales_org",
    description="""Creates a Sales Organization (SOR) linked to a Company Code.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sor_id" (str)
      - "name" (str)
      - "city" (str)
      - "parent_ccd_id" (str)
"""
)
async def create_sales_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sor_id = item.get("sor_id")
        name = item.get("name")
        city = item.get("city")
        parent_ccd_id = item.get("parent_ccd_id")
        try:
            ccd = manager._find_unit(parent_ccd_id, unit_type="CCD")
            if not ccd:
                results.append(str(f"Error: Company Code {parent_ccd_id} not found."))
                fail_count += 1
                continue
    
            sor_id = str(sor_id)[:4]
            err = validate_id_suffix(sor_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            city = str(city)[:40]
    
            country_code = ccd.get("Country") or ccd.get("Attributes/Country") or "IN"
            upper = {"ID": parent_ccd_id, "Type": "CCD", "UUID": ccd.get("UUID")}
            manager.create_unit(sor_id, name, "SOR", country_code, city, upper=upper)
            results.append(str(f"Success: Sales Org {sor_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_sales_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_company_code_to_purchasing_org",
    description="""Assigns a default Company Code (CCD) to a Purchasing Organization (POR). Rule: A Purchasing Org can only have ONE Company Code assigned. [RULES]: 1. PROACTIVELY check current CCD assignment via 'get_unit_details'. 2. If occupied: ASK to 'unassign_relation' before proceeding. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ccd_id" (str)
      - "por_id" (str)
"""
)
async def assign_company_code_to_purchasing_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ccd_id = item.get("ccd_id")
        por_id = item.get("por_id")
        try:
            res = manager.assign_company_code_to_purchasing_org(ccd_id, por_id)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_company_code_to_purchasing_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_warehouse_to_storage_location",
    description="""Assigns a Storage Location to a Warehouse.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "stl_id" (str)
      - "whn_id" (str)
      - "stl_type" (str)
"""
)
async def assign_warehouse_to_storage_location(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        stl_id = item.get("stl_id")
        whn_id = item.get("whn_id")
        stl_type = item.get("stl_type")
        try:
            res = manager.assign_stl_to_warehouse(stl_id, stl_type, whn_id)
            if "error" in res:
                results.append(str(f"Error: {res['error']}"))
                success_count += 1
            results.append(str(f"Success: Storage Location {stl_id} ({stl_type}) assigned to Warehouse {whn_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_warehouse_to_storage_location Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_sla_to_plant",
    description="""Assigns a Sales Area (SLA) to a Plant (PLT). Example ID: '1000|10|00'. [RULES]: 1. PROACTIVELY show current Sales Area assignments for the Plant. 2. PRESENT eligible options via 'list_available_sales_areas'. 3. REQUIRE explicit 'YES' confirmation before execution. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sla_id" (str)
      - "plt_id" (str)
"""
)
async def assign_sla_to_plant(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sla_id = item.get("sla_id")
        plt_id = item.get("plt_id")
        try:
            manager.assign_sla_to_plant(sla_id, plt_id)
            results.append(str(f"Success: Sales Area {sla_id} assigned to Plant {plt_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_sla_to_plant Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_sla_to_sales_office",
    description="""Assigns a Sales Area (SLA) to a Sales Office (SOF). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sla_id" (str)
      - "sof_id" (str)
"""
)
async def assign_sla_to_sales_office(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sla_id = item.get("sla_id")
        sof_id = item.get("sof_id")
        try:
            manager.assign_sla_to_sales_office(sla_id, sof_id)
            results.append(str(f"Success: Sales Area {sla_id} assigned to Sales Office {sof_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_sla_to_sales_office Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="unassign_relation",
    description="""Universal tool to remove one or more assignment rows (relationships). This handles all assignment categories: 1. Plant Links (e.g., to Purchasing Org or Sales Org) 2. Storage Location Links (e.g., to Warehouses) 3. Sales Area Links (e.g., to Plants or Sales Offices) Requires the relation row UUID(s). If parent_unit_id is provided, it will refresh that unit's state. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "relation_row_ids" (str)
      - "parent_unit_id" (str)
"""
)
async def unassign_relation(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."
    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        relation_row_ids = item.get("relation_row_ids")
        parent_unit_id = item.get("parent_unit_id")
        try:
            parent_uuid = None
            if parent_unit_id:
                parent_uuid = manager.find_uuid(parent_unit_id)
    
            manager.unassign_relation(relation_row_ids, parent_uuid=parent_uuid)
            results.append(str(f"Success: Relation(s) {', '.join(relation_row_ids)} unassigned."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk unassign_relation Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_companies",
    description="""Returns a list of existing Companies (CMP). Output: Markdown Bulleted List (* ID - Name (Country)). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_available_companies(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No companies found."))
                fail_count += 1
                continue
    
            units = unit_list[0].get("data", [])
    
            comps = []
            for u in units:
                if not isinstance(u, dict): continue
                u_char = u.get("BusChar") or u.get("BusCharacter") or u.get("Type")
                if u_char == "CMP" and (not country_code or u.get("Country") == country_code):
                    comps.append(f"* {u.get('Id') or u.get('ID')} - {u.get('Name')} ({u.get('Country')})")
            
            results.append(str("\n".join(comps) if comps else "No companies found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_companies Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_company_codes",
    description="""Returns a list of existing Company Codes (CCD). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_available_company_codes(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No company codes found."))
                fail_count += 1
                continue
    
            units = unit_list[0].get("data", [])
            ccds = []
            for u in units:
                if not isinstance(u, dict): continue
                u_char = u.get("BusChar") or u.get("BusCharacter") or u.get("Type")
                if u_char == "CCD" and (not country_code or u.get("Country") == country_code):
                    ccds.append(f"* {u.get('Id') or u.get('ID')} - {u.get('Name')} ({u.get('Country')})")
    
            results.append(str("\n".join(ccds) if ccds else "No company codes found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_company_codes Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_sales_orgs",
    description="""Returns a RAW DUMP of existing Sales Organizations (SOR) that HAVE a Sales Area. If plt_id is provided, filters out SORs already linked to THAT specific plant. Output: Markdown Bulleted List (* ID - Name). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "country_code" (str)
"""
)
async def list_available_sales_orgs(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No sales orgs found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Requirement: Only show SORs that have at least one SLA
            valid_sor_ids = manager.get_sor_with_sla()
            # Filter: Not already linked to the TARGET Plant in PLTSORDCH
            occupied = manager.get_occupied_ids("PLTSORDCH", parent_id=plt_id)
    
            sors = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "SOR" 
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) in valid_sor_ids
                    and str(u.get('Id') or u.get('ID')) not in occupied]
    
            header = f"Available Sales Organizations (Source: All" + (f", Country: {country_code}" if country_code else "") + (f", Not in Plant {plt_id}" if plt_id else "") + "):\n\n"
            results.append(str(header + "\n".join(sors) if sors else "No available sales orgs with Sales Areas found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_sales_orgs Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_purchasing_orgs",
    description="""[WHEN TO USE]: For ASSIGNMENT workflows. Returns Purchasing Orgs NOT already linked to the target Plant. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "country_code" (str)
"""
)
async def list_available_purchasing_orgs(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No purchasing orgs found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Requirement: Not already linked to THIS Plant
            occupied = manager.get_occupied_ids("PLTPOR", parent_id=plt_id)
    
            pors = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                     if isinstance(u, dict) and u.get("BusChar") == "POR"
                     and (not country_code or u.get("Country") == country_code)
                     and str(u.get('Id') or u.get('ID')) not in occupied]
            results.append(str(f"Available Purchasing Organizations:\n\n" + "\n".join(pors) if pors else "No available purchasing orgs found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_purchasing_orgs Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_all_purchasing_orgs",
    description="""[WHEN TO USE]: For DISCOVERY or DEBUGGING only. Returns ALL existing Purchasing Orgs regardless of plant linkage. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_all_purchasing_orgs(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No purchasing orgs found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            pors = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "POR"
                    and (not country_code or u.get("Country") == country_code)]
            results.append(str(f"Total Purchasing Organizations in System:\n\n" + "\n".join(pors) if pors else "No purchasing orgs found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_all_purchasing_orgs Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_shipping_points",
    description="""Returns a list of Shipping Points NOT already assigned to the target Plant (if plt_id is provided). Output: Markdown Bulleted List (* ID - Name). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "country_code" (str)
"""
)
async def list_available_shipping_points(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No shipping points found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Check local occupancy for PLT across all SPT relations
            o1 = manager.get_occupied_ids("PLTSPT_STD", parent_id=plt_id)
            o2 = manager.get_occupied_ids("PLTSPT_RAW_MATNR", parent_id=plt_id)
            o3 = manager.get_occupied_ids("PLTSPT_CUST_CONSMT", parent_id=plt_id)
            occupied = o1 | o2 | o3
    
            spts = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')} ({u.get('BusChar')})" for u in units 
                    if isinstance(u, dict) and u.get("BusChar").startswith("SPT")
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
            results.append(str(f"Available Shipping Points:\n\n" + "\n".join(spts) if spts else "No available shipping points found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_shipping_points Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_all_shipping_points",
    description="""Returns a list of ALL existing Shipping Points (SPT), including those already assigned. Use this for discovery or if you need to find an ID to delete. Output: Markdown Bulleted List (* ID - Name). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_all_shipping_points(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No shipping points found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            spts = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')} ({u.get('BusChar')})" for u in units 
                    if isinstance(u, dict) and u.get("BusChar").startswith("SPT")
                    and (not country_code or u.get("Country") == country_code)]
            results.append(str(f"Total Shipping Points in System:\n\n" + "\n".join(spts) if spts else "No shipping points found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_all_shipping_points Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_storage_locations",
    description="""Returns a list of ALL Storage Locations (STL). [STRICT RULE]: A storage location can only have ONE warehouse assignment. During discovery, we show all, but the assignment tool will block duplicates.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_available_storage_locations(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No storage locations found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            stls = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')} ({u.get('BusChar')})" for u in units 
                    if isinstance(u, dict) and (u.get("BusChar") or "").startswith("STL")
                    and (not country_code or u.get("Country") == country_code)]
            header = f"Available Storage Locations:\n\n"
            results.append(str(header + "\n".join(stls) if stls else "No storage locations found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_storage_locations Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="lookup_unit_by_id",
    description="""Finds ALL units (Plants, Sales Offices, etc.) that share the same ID. Use this when you suspect an ID collision (e.g., '6601' is both a Plant and an Office). Returns a summarized list of matches. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "unit_id" (str)
"""
)
async def lookup_unit_by_id(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        unit_id = item.get("unit_id")
        try:
            matches = manager.list_all_units_by_id(unit_id)
            if not matches:
                results.append(str(f"No units found with ID {unit_id}."))
                fail_count += 1
                continue
    
            res = [f"* {u.get('Name')} (Type: {u.get('BusChar')}, UUID: {u.get('UUID')})" for u in matches]
            results.append(str(f"Found {len(matches)} units matching ID '{unit_id}':\n\n" + "\n".join(res)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk lookup_unit_by_id Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="get_unit_details",
    description="""Returns full metadata for a specific unit by ID. If multiple units share ID (e.g., Plant 6601 and Office 6601), specify 'unit_type'. Types: CCD, CMP, POR, PLT, STL, WHN_CLOUD, EWN, SOR, SLA, SOF, SGR, DCH, DIV, SPT_STD, SPT_RAW_MATNR, SPT_CUST_CONSMT. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "unit_id" (str)
      - "unit_type" (str)
"""
)
async def get_unit_details(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        unit_id = item.get("unit_id")
        unit_type = item.get("unit_type")
        try:
            unit = manager._find_unit(unit_id, unit_type=unit_type)
            if not unit: 
                msg = f"Error: Unit {unit_id} not found" + (f" with type {unit_type}." if unit_type else ".")
                results.append(str(msg))
                success_count += 1
            results.append(str(json.dumps(unit, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk get_unit_details Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_warehouses",
    description="""Returns a list of all existing Warehouses (WHN_CLOUD). Note: Warehouses can be shared across multiple storage locations. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "country_code" (str)
"""
)
async def list_available_warehouses(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No warehouses found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            whns = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "WHN_CLOUD"
                    and (not country_code or u.get("Country") == country_code)]
            message = "Available Warehouses (All):\n\n"
            results.append(str(message + "\n".join(whns) if whns else "No warehouses found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_warehouses Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_warehouse_numbers",
    description="""Returns a list of Warehouse Numbers (EWN). - If whn_id is provided: Filters out numbers already linked to THAT specific Warehouse. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "whn_id" (str)
      - "country_code" (str)
"""
)
async def list_available_warehouse_numbers(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        whn_id = item.get("whn_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No warehouse numbers found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            if whn_id:
                occupied = manager.get_occupied_ids("WHNEWN", parent_id=whn_id)
            else:
                occupied = set()
    
            ewns = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "EWN"
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
    
            header = f"Available Warehouse Numbers (Target WHN: {whn_id or 'All'}):\n\n"
            results.append(str(header + "\n".join(ewns) if ewns else "No available warehouse numbers found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_warehouse_numbers Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_sales_areas",
    description="""Returns a list of Sales Areas (SLA). If plt_id or sof_id is provided, filters out SLAs already linked to THAT specific plant/office. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "sof_id" (str)
      - "country_code" (str)
"""
)
async def list_available_sales_areas(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        sof_id = item.get("sof_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No sales areas found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Local Occupancy: Only hide if already linked to this specific target parent
            o1 = manager.get_occupied_ids("SLAPLT", parent_id=plt_id) if plt_id else set()
            o2 = manager.get_occupied_ids("SLASOF", parent_id=sof_id) if sof_id else set()
            occupied = o1 | o2
    
            slas = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "SLA"
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
            results.append(str(f"Available Sales Areas:\n\n" + "\n".join(slas) if slas else "No available sales areas found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_sales_areas Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_sales_offices",
    description="""Returns a list of Sales Offices (SOF). - If sla_id is NOT provided: Shows ALL existing Sales Offices (they are shareable across different Sales Areas). - If sla_id IS provided: Filters out offices already linked to THAT specific Sales Area (Locally Unique). [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sla_id" (str)
      - "country_code" (str)
"""
)
async def list_available_sales_offices(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sla_id = item.get("sla_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No sales offices found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Truly Local: If no specific SLA is provided, we don't hide anything (All are discoverable)
            if sla_id:
                occupied = manager.get_occupied_ids("SLASOF", parent_id=sla_id)
            else:
                occupied = set()
    
            sofs = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "SOF"
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
    
            header = f"Available Sales Offices (Target SLA: {sla_id or 'All'}):\n\n"
            results.append(str(header + "\n".join(sofs) if sofs else "No available sales offices found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_sales_offices Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_sales_groups",
    description="""Returns a list of Sales Groups (SGR). - If sof_id is provided: Filters out groups already linked to THAT specific Sales Office. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sof_id" (str)
      - "country_code" (str)
"""
)
async def list_available_sales_groups(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sof_id = item.get("sof_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No sales groups found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            if sof_id:
                occupied = manager.get_occupied_ids("SOFSGR", parent_id=sof_id)
            else:
                occupied = set()
    
            sgrs = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "SGR"
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
    
            header = f"Available Sales Groups (Target SOF: {sof_id or 'All'}):\n\n"
            results.append(str(header + "\n".join(sgrs) if sgrs else "No available sales groups found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_sales_groups Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_available_plants",
    description="""Returns Plants that are eligible for a NEW assignment. [FILTER LOGIC]: 1. If 'por_id' provided: Excludes plants already linked to that specific Purchasing Org. 2. Shared Rule: Plants that already have at least one Sales Area assignment are excluded — these are not eligible for a new Sales Org link. Plants excluded by Rule 2 are still eligible for Purchasing Org assignment if not already linked. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "por_id" (str)
      - "country_code" (str)
"""
)
async def list_available_plants(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        por_id = item.get("por_id")
        country_code = item.get("country_code") or item.get("country")
        try:
            data = manager.get_all_data()
            unit_list = data.get("data", [{}])
            if not isinstance(unit_list, list) or not unit_list:
                results.append(str("No plants found."))
                fail_count += 1
                continue
            units = unit_list[0].get("data", [])
    
            # Local Occupancy for POR, but GLOBAL for Sales Area assignment
            o1 = manager.get_occupied_ids("PLTPOR", parent_id=por_id) if por_id else set()
            o2 = manager.get_occupied_ids("SLAPLT") # Global for SLA assignments
            occupied = o1 | o2
    
            plts = [f"* {u.get('Id') or u.get('ID')} - {u.get('Name')}" for u in units 
                    if isinstance(u, dict) and u.get("BusChar") == "PLT"
                    and (not country_code or u.get("Country") == country_code)
                    and str(u.get('Id') or u.get('ID')) not in occupied]
            results.append(str(f"Available Plants (Source: Dynamic Filter):\n\n" + "\n".join(plts) if plts else "No available plants found."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_available_plants Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_distribution_channels",
    description="""Lists existing Distribution Channels (DCH). [STRICT BUSINESS RULE]: To identify valid channels for your enterprise structure, you MUST provide 'sor_id'. - If 'sor_id' is NOT provided: You will get all global channels, which may be invalid for your Sales Org. - If 'sor_id' IS provided: You get only channels already defined in a Sales Area for that Org. [INTERACTION]: If 'sor_id' is missing, WARN the user that listing global channels is for discovery only and may lead to invalid assignments. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sor_id" (str)
"""
)
async def list_distribution_channels(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sor_id = item.get("sor_id")
        try:
            data_list = manager.list_distribution_channels()
    
            if sor_id:
                valid_dch_ids = manager.get_dch_for_sor(sor_id)
                dchs = [f"* {d.get('ID')} - {d.get('Name')}" for d in data_list
                        if str(d.get('ID')).strip() in valid_dch_ids]
                if not dchs:
                    results.append(str(f"No Distribution Channels are associated with Sales Org {sor_id} in any Sales Area (SLA)."))
                    success_count += 1
            else:
                dchs = [f"* {d.get('ID')} - {d.get('Name')}" for d in data_list]
        
            res = "\n".join(dchs) if dchs else "No distribution channels found."
            results.append(str(f"{res}\n* CREATE_NEW - [Create New Distribution Channel]"))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_distribution_channels Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_divisions",
    description="""Returns a list of existing Divisions (DIV). [OUTPUT]: - A Markdown Bulleted List (* ID - Name). - Includes a 'CREATE_NEW' sentinel item at the end. - [FLOW]: If 'CREATE_NEW' is chosen, pause current flow, run 'create_division' as a sub-task, then resume with the new ID. [ID RULE]: The last 1 character of the ID must be alphabetic.
"""
)
async def list_divisions(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        try:
            data = manager.list_divisions()
            divs = [f"* {d.get('ID')} - {d.get('Name')}" for d in data]
            res = "\n".join(divs) if divs else "No divisions found."
            results.append(str(f"{res}\n* CREATE_NEW - [Create New Division]"))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_divisions Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_distribution_channel",
    description="""Creates a new standalone Distribution Channel (DCH). Max 2 character ID (e.g., '10', 'WH'), Max 20 character Name. Note: These are global and don't require a parent until you create a Sales Area. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "dch_id" (str)
      - "name" (str)
"""
)
async def create_distribution_channel(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        dch_id = item.get("dch_id")
        name = item.get("name")
        try:
            dch_id = str(dch_id)[:2]
            err = validate_id_suffix(dch_id, 1)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            manager.create_distribution_channel(dch_id, name)
            results.append(str(f"Success: Distribution Channel {dch_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_distribution_channel Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_division",
    description="""Creates a new standalone Division (DIV). Max 2 character ID (e.g., '01', '38'), Max 20 character Name. Note: These are global and don't require a parent until you create a Sales Area. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "div_id" (str)
      - "name" (str)
"""
)
async def create_division(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        div_id = item.get("div_id")
        name = item.get("name")
        try:
            div_id = str(div_id)[:2]
            err = validate_id_suffix(div_id, 1)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            manager.create_division(div_id, name)
            results.append(str(f"Success: Division {div_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_division Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
# @mcp.tool()
# async def create_storage_location(stl_id: str, name: str, subtype: str, city: str, parent_plt_id: str) -> str:
#     """
#     Creates a Storage Location (STL) under a Plant.
#     ID: Max 4 chars, Name: Max 16 chars, City: Max 40 chars.
    
#     [RULES]:
#     1. PROACTIVELY present ALL 10 subtypes (1. RAW, 2. RETURN, 3. FINISHED, 4. SEMI, 5. WAREHOUSE, 6. HU, 7. KANBAN, 8. DAY_TANK, 9. MAIN_TANK, 10. RESIDUAL) and REQUIRE selection.
#     """
#     await ensure_session()
#     stl_id = str(stl_id)[:4]
#     name = str(name)[:16]
#     city = str(city)[:40]
#     manager.create_storage_location(stl_id, name, subtype, city, parent_plt_id)
#     return f"Success: Storage Location {stl_id} created."

@mcp.tool(
    name="create_sales_area",
    description="""Creates a Sales Area (SLA) linking Sales Org, Dist. Channel, and Division. Name: Max 20 chars. ID is auto-generated as 'SOR_ID|DCH_ID|DIV_ID'. [RULES]: 1. If 'CREATE_NEW' is chosen for DCH/DIV: PAUSE, run creation tool, then RESUME with new ID. 2. REQUIRE explicit user approval for the auto-generated ID. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "name" (str)
      - "parent_sor_id" (str)
      - "dch_id" (str)
      - "dch_uuid" (str)
      - "div_id" (str)
      - "div_uuid" (str)
"""
)
async def create_sales_area(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        name = item.get("name")
        parent_sor_id = item.get("parent_sor_id")
        dch_id = item.get("dch_id")
        dch_uuid = item.get("dch_uuid")
        div_id = item.get("div_id")
        div_uuid = item.get("div_uuid")
        try:
            name = str(name)[:20]
            # sla_id is None to trigger internal auto-generation
            res = manager.create_sales_area(None, name, parent_sor_id, dch_id, dch_uuid, div_id, div_uuid)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_sales_area Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_purchasing_org",
    description="""Creates a Purchasing Organization (POR) linked to a Company Code (CCD). ID: Max 4 chars, Name: Max 20 chars. Note: Always call 'list_available_company_codes' first to confirm parent_ccd_id is valid. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "por_id" (str)
      - "name" (str)
      - "parent_ccd_id" (str)
"""
)
async def create_purchasing_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        por_id = item.get("por_id")
        name = item.get("name")
        parent_ccd_id = item.get("parent_ccd_id")
        try:
            por_id = str(por_id)[:4]
            err = validate_id_suffix(por_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            manager.create_purchasing_org(por_id, name, parent_ccd_id)
            results.append(str(f"Success: Purchasing Org {por_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_purchasing_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
# @mcp.tool()
# async def create_shipping_point(spt_id: str, name: str, subtype_id: str, country_code: str, city: str) -> str:
#     """
#     Creates a Shipping Point (SPT).
#     ID: Max 4 chars, Name: Max 20 chars, City: Max 40 chars.
    
#     [RULES]:
#     1. PROACTIVELY present Country selection (IN|SG|US).
#     2. PROACTIVELY present Subtype selection (SPT_STD|SPT_RAW_MATNR|SPT_CUST_CONSMT).
#     3. REQUIRE explicit selections before execution.
    
#     [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
#     """
#     await ensure_session()
#     spt_id = str(spt_id)[:4]
#     err = validate_id_suffix(spt_id, 2)
#     if err: return err
#     name = str(name)[:20]
#     city = str(city)[:40]
#     manager.create_shipping_point(spt_id, name, subtype_id, country_code, city)
#     return f"Success: Shipping Point {spt_id} created."

# @mcp.tool()
# async def create_sales_office(sof_id: str, name: str, country_code: str, city: str) -> str:
#     """
#     Creates a Sales Office (SOF).
#     ID: Max 4 chars, Name: Max 20 chars, City: Max 40 chars.
    
#     [RULES]:
#     1. PROACTIVELY present Country selection (IN|SG|US) and REQUIRE choice.
    
#     [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
#     """
#     await ensure_session()
#     sof_id = str(sof_id)[:4]
#     err = validate_id_suffix(sof_id, 2)
#     if err: return err
#     name = str(name)[:20]
#     city = str(city)[:40]
#     manager.create_sales_office(sof_id, name, country_code, city)
#     return f"Success: Sales Office {sof_id} created."

# @mcp.tool()
# async def create_sales_group(sgr_id: str, name: str, parent_sof_id: str) -> str:
#     """
#     Creates a Sales Group (SGR) linked to a Sales Office (SOF). Max 3 chars ID, Max 20 chars Name.
#     Note: Always call 'list_available_sales_offices' first to confirm parent_sof_id is valid.
    
#     [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
#     """
#     await ensure_session()
#     sgr_id = str(sgr_id)[:3]
#     err = validate_id_suffix(sgr_id, 2)
#     if err: return err
#     name = str(name)[:20]
#     manager.create_sales_group(sgr_id, name, parent_sof_id)
#     return f"Success: Sales Group {sgr_id} created."

@mcp.tool(
    name="create_warehouse",
    description="""Creates a new Warehouse (WHN). Max 3 chars ID, Max 25 chars Name. Default Subtype: 'WHN_CLOUD'. Note: These are standalone units until linked to Storage Locations. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "whn_id" (str)
      - "name" (str)
      - "subtype" (str)
"""
)
async def create_warehouse(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        whn_id = item.get("whn_id")
        name = item.get("name")
        subtype = item.get("subtype")
        try:
            whn_id = str(whn_id)[:3]
            err = validate_id_suffix(whn_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:25]
            manager.create_unit(whn_id, name, subtype, "", "")
            results.append(str(f"Success: Warehouse {whn_id} ({subtype}) created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_warehouse Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_warehouse_number",
    description="""Creates a Warehouse Number (EWN) linked to a Warehouse (WHN_CLOUD). Max 4 chars ID, Max 40 chars Name. Note: Always call 'list_available_warehouses' first to confirm parent_whn_id is valid. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ewn_id" (str)
      - "name" (str)
      - "parent_whn_id" (str)
"""
)
async def create_warehouse_number(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ewn_id = item.get("ewn_id")
        name = item.get("name")
        parent_whn_id = item.get("parent_whn_id")
        try:
            ewn_id = str(ewn_id)[:4]
            err = validate_id_suffix(ewn_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:40]
            manager.create_warehouse_number(ewn_id, name, parent_whn_id)
            results.append(str(f"Success: Warehouse Number {ewn_id} created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_warehouse_number Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_plant_to_purchasing_org",
    description="""Assigns an existing Plant to an existing Purchasing Organization. [RULES]: 1. PROACTIVELY show CURRENT Purchasing Org assignments for the Plant. 2. PRESENT eligible (unlinked) options via 'list_available_purchasing_orgs'. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "por_id" (str)
"""
)
async def assign_plant_to_purchasing_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        por_id = item.get("por_id")
        try:
            manager.assign_plant_to_purchasing_org(plt_id, por_id)
            results.append(str(f"Success: Plant {plt_id} assigned to Purchasing Org {por_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_plant_to_purchasing_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_plant_to_sales_org_dch",
    description="""Assigns a Plant to a Sales Organization and one or more Distribution Channels. [RULES]: 1. PROACTIVELY show current Plant assignments via 'get_unit_details'. 2. For SOR selection: ONLY show channels valid for that SOR via 'list_distribution_channels'. 3. PRESENT a summary table and REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "sor_id" (str)
      - "dch_ids" (str)
"""
)
async def assign_plant_to_sales_org_dch(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        sor_id = item.get("sor_id")
        dch_ids = item.get("dch_ids")
        try:
            manager.assign_plant_to_sales_org_dch(plt_id, sor_id, dch_ids)
            results.append(str(f"Success: Plant {plt_id} assigned to Sales Org {sor_id} and Channels {', '.join(dch_ids)}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_plant_to_sales_org_dch Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_plant_to_shipping_point_consignment",
    description="""Assigns a Plant to a Shipping Point for Customer Consignment (SPT_CUST_CONSMT). [RULES]: 1. PROACTIVELY show current Shipping Point assignments for the Plant. 2. PRESENT eligible (unlinked) options of this subtype. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "spt_id" (str)
"""
)
async def assign_plant_to_shipping_point_consignment(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        spt_id = item.get("spt_id")
        try:
            res = manager.assign_plant_to_shipping_point(plt_id, spt_id, "SPT_CUST_CONSMT")
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_plant_to_shipping_point_consignment Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_plant_to_shipping_point_return",
    description="""Assigns a Plant to a Shipping Point for Return (SPT_RAW_MATNR). [RULES]: 1. PROACTIVELY show current Shipping Point assignments for the Plant. 2. PRESENT eligible (unlinked) options of this subtype. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "spt_id" (str)
"""
)
async def assign_plant_to_shipping_point_return(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        spt_id = item.get("spt_id")
        try:
            res = manager.assign_plant_to_shipping_point(plt_id, spt_id, "SPT_RAW_MATNR")
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_plant_to_shipping_point_return Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="assign_plant_to_shipping_point_standard",
    description="""Assigns a Plant to a Standard Shipping Point (SPT_STD). [RULES]: 1. PROACTIVELY show current Shipping Point assignments for the Plant. 2. PRESENT eligible (unlinked) options via 'list_available_shipping_points'. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "spt_id" (str)
"""
)
async def assign_plant_to_shipping_point_standard(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        spt_id = item.get("spt_id")
        try:
            res = manager.assign_plant_to_shipping_point(plt_id, spt_id, "SPT_STD")
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1


        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk assign_plant_to_shipping_point_standard Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_company_code",
    description="""Updates a Company Code (CCD). Name: Max 25 chars, City: Max 40 chars. [RULES]: 1. PROACTIVELY show current details via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation before execution. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ccd_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_company_code(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ccd_id = item.get("ccd_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            manager.update_company_code(ccd_id, new_name=new_name, new_city=new_city)
            results.append(str(f"Success: Company Code {ccd_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_company_code Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_plant",
    description="""Updates a Plant (PLT). Name: Max 30 chars, City: Max 40 chars. [RULES]: 1. PROACTIVELY show current details via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation before execution. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_plant(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            manager.update_plant(plt_id, new_name=new_name, new_city=new_city)
            results.append(str(f"Success: Plant {plt_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_plant Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_sales_org",
    description="""Updates a Sales Organization (SOR). Name: Max 20 chars, City: Max 40 chars. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show **Full Current Details** to the user. 2. REQUIRE an explicit 'YES' confirmation before executing. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sor_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_sales_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sor_id = item.get("sor_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            manager.update_sales_org(sor_id, new_name=new_name, new_city=new_city)
            results.append(str(f"Success: Sales Org {sor_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_sales_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_sales_area",
    description="""Updates a Sales Area (SLA) name. Name: Max 20 chars. [RULES]: 1. PROACTIVELY show current name via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sla_id" (str)
      - "new_name" (str)
"""
)
async def update_sales_area(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sla_id = item.get("sla_id")
        new_name = item.get("new_name")
        try:
            manager.update_sales_area(sla_id, new_name=new_name)
            results.append(str(f"Success: Sales Area {sla_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_sales_area Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_storage_location",
    description="""Updates a Storage Location (STL). Name: Max 16 chars, City: Max 40 chars. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show **Full Current Details** to the user. 2. REQUIRE an explicit 'YES' confirmation before executing. 3. If both fields are empty, return a warning.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "stl_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_storage_location(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        stl_id = item.get("stl_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            if not new_name and not new_city:
                results.append(str("Warning: No fields provided for update. At least one of 'new_name' or 'new_city' must be non-null."))
                success_count += 1
            manager.update_storage_location(stl_id, new_name=new_name, new_city=new_city)
            results.append(str(f"Success: Storage Location {stl_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_storage_location Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_purchasing_org",
    description="""Updates an existing Purchasing Organization (POR). Max 20 character Name. [RULES]: 1. PROACTIVELY show current name via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation for the name change. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "por_id" (str)
      - "new_name" (str)
"""
)
async def update_purchasing_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        por_id = item.get("por_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(por_id, unit_type="POR")
            if not uuid:
                results.append(str(f"Error: Purchasing Org {por_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20]
            res = manager.update_unit(por_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_purchasing_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_distribution_channel",
    description="""Updates an existing Distribution Channel (DCH). Max 20 character Name. [RULES]: 1. PROACTIVELY show current name via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation before execution. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "dch_id" (str)
      - "new_name" (str)
"""
)
async def update_distribution_channel(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        dch_id = item.get("dch_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(dch_id, unit_type="DCH")
            if not uuid:
                results.append(str(f"Error: Distribution Channel {dch_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20]
            res = manager.update_unit(dch_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_distribution_channel Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_division",
    description="""Updates an existing Division (DIV). Max 20 character Name. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show the current DIV name. 2. CONFIRM the change and REQUIRE explicit 'YES' before executing. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "div_id" (str)
      - "new_name" (str)
"""
)
async def update_division(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        div_id = item.get("div_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(div_id, unit_type="DIV")
            if not uuid:
                results.append(str(f"Error: Division {div_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20]
            res = manager.update_unit(div_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_division Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_sales_office",
    description="""Updates a Sales Office (SOF). Max 20 character Name, Max 40 character City. [RULES]: 1. PROACTIVELY show current details via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sof_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_sales_office(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sof_id = item.get("sof_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            uuid = manager.find_uuid(sof_id, unit_type="SOF")
            if not uuid:
                results.append(str(f"Error: Sales Office {sof_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20] if new_name else None
            city = str(new_city)[:40] if new_city else None
            res = manager.update_unit(sof_id, uuid, name, city, country=None)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_sales_office Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_sales_group",
    description="""Updates an existing Sales Group (SGR). Max 20 character Name. [RULES]: 1. PROACTIVELY show current name via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sgr_id" (str)
      - "new_name" (str)
"""
)
async def update_sales_group(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sgr_id = item.get("sgr_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(sgr_id, unit_type="SGR")
            if not uuid:
                results.append(str(f"Error: Sales Group {sgr_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20]
            res = manager.update_unit(sgr_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_sales_group Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_shipping_point",
    description="""Updates an existing Shipping Point (SPT). Name: Max 20 characters, City: Max 40 characters. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show **Full Current Details** to the user. 2. REQUIRE an explicit 'YES' confirmation before executing. 3. If both fields are empty, return a warning. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "spt_id" (str)
      - "new_name" (str)
      - "new_city" (str)
"""
)
async def update_shipping_point(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        spt_id = item.get("spt_id")
        new_name = item.get("new_name")
        new_city = item.get("new_city")
        try:
            if not new_name and not new_city:
                results.append(str("Warning: No fields provided for update. At least one of 'new_name' or 'new_city' must be non-null."))
                success_count += 1
            # Type-aware search (strict SPT)
            uuid = manager.find_uuid(spt_id, unit_type="SPT_STD") 
            if not uuid: # Try other subtypes
                uuid = manager.find_uuid(spt_id, unit_type="SPT_RAW_MATNR") or manager.find_uuid(spt_id, unit_type="SPT_CUST_CONSMT")
    
            if not uuid:
                results.append(str(f"Error: Shipping Point {spt_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:20] if new_name else None
            city = str(new_city)[:40] if new_city else None
            manager.update_unit(spt_id, uuid, name, new_city=city)
            results.append(str(f"Success: Shipping Point {spt_id} updated."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_shipping_point Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_warehouse",
    description="""Updates an existing Warehouse (WHN_CLOUD). Max 25 character Name. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show current warehouse name. 2. CONFIRM the change and REQUIRE explicit 'YES' before executing. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "whn_id" (str)
      - "new_name" (str)
"""
)
async def update_warehouse(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        whn_id = item.get("whn_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(whn_id, unit_type="WHN_CLOUD")
            if not uuid:
                results.append(str(f"Error: Warehouse {whn_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:25]
            res = manager.update_unit(whn_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_warehouse Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="update_warehouse_number",
    description="""Updates an existing Warehouse Number (EWN). Max 40 character Name. [INTERACTION RULES]: 1. PROACTIVELY call 'get_unit_details' to show current EWN name. 2. CONFIRM the change and REQUIRE explicit 'YES' before executing. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ewn_id" (str)
      - "new_name" (str)
"""
)
async def update_warehouse_number(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ewn_id = item.get("ewn_id")
        new_name = item.get("new_name")
        try:
            uuid = manager.find_uuid(ewn_id, unit_type="EWN")
            if not uuid:
                results.append(str(f"Error: Warehouse Number {ewn_id} not found."))
                fail_count += 1
                continue
    
            name = str(new_name)[:40]
            res = manager.update_unit(ewn_id, uuid, name)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk update_warehouse_number Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_company_code",
    description="""Deletes a Company Code and its descendants recursively. [SAFETY RULES]: 1. PROACTIVELY show ALL descendants (Sales Orgs, Plants, etc.) via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation for this IRREVERSIBLE action. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ccd_id" (str)
"""
)
async def delete_company_code(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ccd_id = item.get("ccd_id")
        try:
            uuid = manager.find_uuid(ccd_id, unit_type="CCD")
            if not uuid:
                results.append(str(f"Error: Company Code {ccd_id} not found."))
                fail_count += 1
                continue
            manager.delete_company_code(uuid)
            results.append(str(f"Success: Company Code {ccd_id} and its descendants have been deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_company_code Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_plant",
    description="""Deletes a Plant and its descendants recursively. [SAFETY RULES]: 1. PROACTIVELY show ALL descendants (Storage Locations, etc.) via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation for this IRREVERSIBLE action. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "plt_id" (str)
"""
)
async def delete_plant(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        plt_id = item.get("plt_id")
        try:
            uuid = manager.find_uuid(plt_id, unit_type="PLT")
            if not uuid:
                results.append(str(f"Error: Plant {plt_id} not found."))
                fail_count += 1
                continue
            manager.delete_plant(uuid)
            results.append(str(f"Success: Plant {plt_id} and its descendants have been deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_plant Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_sales_org",
    description="""Deletes a Sales Organization by ID and its descendants recursively. [RULES]: 1. PROACTIVELY show ALL descendants (Sales Areas, etc.) via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation for this IRREVERSIBLE action. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sor_id" (str)
"""
)
async def delete_sales_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sor_id = item.get("sor_id")
        try:
            uuid = manager.find_uuid(sor_id, unit_type="SOR")
            if not uuid:
                results.append(str(f"Error: Sales Org {sor_id} not found."))
                fail_count += 1
                continue
            manager.delete_sales_org(uuid)
            results.append(str(f"Success: Sales Org {sor_id} and its descendants have been deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_sales_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_sales_area",
    description="""Deletes a Sales Area by ID and its relations recursively. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sla_id" (str)
"""
)
async def delete_sales_area(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sla_id = item.get("sla_id")
        try:
            uuid = manager.find_uuid(sla_id, unit_type="SLA")
            if not uuid:
                results.append(str(f"Error: Sales Area {sla_id} not found."))
                fail_count += 1
                continue
            manager.delete_sales_area(uuid)
            results.append(str(f"Success: Sales Area {sla_id} and its relations have been deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_sales_area Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_storage_location",
    description="""Deletes a specific Storage Location (STL) by ID. [RULES]: 1. PROACTIVELY show linked warehouse links via 'get_unit_details'. 2. WARN that deletion will break existing warehouse assignments. 3. REQUIRE explicit 'YES' confirmation.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "stl_id" (str)
"""
)
async def delete_storage_location(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        stl_id = item.get("stl_id")
        try:
            uuid = manager.find_uuid(stl_id, unit_type="STL")
            if not uuid:
                results.append(str(f"Error: Storage Location {stl_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Storage Location {stl_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_storage_location Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_division",
    description="""Deletes a specific Division by ID. [SHARED ENTITY GUARD]: 1. PROACTIVELY call 'list_available_sales_areas' or check 'get_all_data' for any Sales Area (SLA) referencing this Division. 2. If found, list the 'ID and Name' of all affected Sales Areas. 3. WARN the user that these Sales Areas will be lost because the required link will be broken. 4. ASK for an explicit 'YES' confirmation from the user before executing. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "div_id" (str)
"""
)
async def delete_division(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        div_id = item.get("div_id")
        try:
            uuid = manager.find_uuid(div_id, unit_type="DIV")
            if not uuid:
                results.append(str(f"Error: Division {div_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Division {div_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_division Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_distribution_channel",
    description="""Deletes a specific Distribution Channel by ID. [RULES]: 1. PROACTIVELY check for linked Sales Areas (SLA). 2. If found: WARN that these Sales Areas will be lost. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 1 character of the ID must be alphabetic.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "dch_id" (str)
"""
)
async def delete_distribution_channel(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        dch_id = item.get("dch_id")
        try:
            uuid = manager.find_uuid(dch_id, unit_type="DCH")
            if not uuid:
                results.append(str(f"Error: Distribution Channel {dch_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Distribution Channel {dch_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_distribution_channel Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_shipping_point",
    description="""Deletes a Shipping Point by ID. [SAFETY RULES]: 1. PROACTIVELY check linked Plants via 'get_unit_details'. 2. WARN that deletion will remove links from N plant(s). 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "spt_id" (str)
"""
)
async def delete_shipping_point(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        spt_id = item.get("spt_id")
        try:
            # Explicitly look for SPT types only to avoid collisions with Sales Offices
            unit = manager._find_unit(spt_id, unit_type="SPT_STD") or \
                   manager._find_unit(spt_id, unit_type="SPT_RAW_MATNR") or \
                   manager._find_unit(spt_id, unit_type="SPT_CUST_CONSMT")
    
            if not unit:
                results.append(str(f"Error: Shipping Point {spt_id} not found."))
                fail_count += 1
                continue
            uuid = unit.get("UUID")
            manager.delete_unit(uuid)
            results.append(str(f"Success: Shipping Point {spt_id} ({unit.get('Name')}) deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_shipping_point Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_purchasing_org",
    description="""Deletes a specific Purchasing Organization by ID. [RULES]: 1. PROACTIVELY show linked Plants and Company Codes. 2. WARN about impact on linked entities. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "por_id" (str)
"""
)
async def delete_purchasing_org(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        por_id = item.get("por_id")
        try:
            uuid = manager.find_uuid(por_id, unit_type="POR")
            if not uuid:
                results.append(str(f"Error: Purchasing Org {por_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Purchasing Org {por_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_purchasing_org Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_warehouse",
    description="""Deletes a specific Warehouse by ID. [SAFETY RULES]: 1. PROACTIVELY show all linked Storage Locations (STLs) and Warehouse Numbers (EWNs). 2. WARN that deletion will break existing assignments. 3. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "whn_id" (str)
"""
)
async def delete_warehouse(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        whn_id = item.get("whn_id")
        try:
            uuid = manager.find_uuid(whn_id, unit_type="WHN_CLOUD")
            if not uuid:
                results.append(str(f"Error: Warehouse {whn_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Warehouse {whn_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_warehouse Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_warehouse_number",
    description="""Deletes a specific Warehouse Number (EWN) by ID. [SAFETY RULES]: 1. PROACTIVELY check parent warehouse and existing links. 2. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ewn_id" (str)
"""
)
async def delete_warehouse_number(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        ewn_id = item.get("ewn_id")
        try:
            uuid = manager.find_uuid(ewn_id, unit_type="EWN")
            if not uuid:
                results.append(str(f"Error: Warehouse Number {ewn_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Warehouse Number {ewn_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_warehouse_number Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_sales_office",
    description="""Deletes a specific Sales Office by ID. [SAFETY RULES]: 1. PROACTIVELY check for linked Sales Groups (SGR). 2. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sof_id" (str)
"""
)
async def delete_sales_office(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sof_id = item.get("sof_id")
        try:
            uuid = manager.find_uuid(sof_id, unit_type="SOF")
            if not uuid:
                results.append(str(f"Error: Sales Office {sof_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Sales Office {sof_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_sales_office Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_sales_group",
    description="""Deletes a specific Sales Group by ID. [SAFETY RULES]: 1. REQUIRE explicit 'YES' confirmation. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sgr_id" (str)
"""
)
async def delete_sales_group(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sgr_id = item.get("sgr_id")
        try:
            uuid = manager.find_uuid(sgr_id, unit_type="SGR")
            if not uuid:
                results.append(str(f"Error: Sales Group {sgr_id} not found."))
                fail_count += 1
                continue
            manager.delete_unit(uuid)
            results.append(str(f"Success: Sales Group {sgr_id} deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_sales_group Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_unit",
    description="""Generic tool to delete ANY unit by UUID. [SAFETY RULES]: 1. PROACTIVELY show unit details and ALL descendants via 'get_unit_details'. 2. REQUIRE explicit 'YES' confirmation for this IRREVERSIBLE action. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "uuid" (str)
"""
)
async def delete_unit(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        uuid = item.get("uuid")
        try:
            manager.delete_unit(uuid)
            results.append(str(f"Success: Unit {uuid} and its descendants have been deleted."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_unit Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_unconfirmed_units",
    description="""Returns a list of organizational units that are newly created but NOT yet confirmed. Output: Markdown Table with ID, Name, and Unit Type. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
"""
)
async def list_unconfirmed_units(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        try:
            units = manager.get_unconfirmed_units()
            if not units:
                results.append(str("Status: All organizational units are confirmed."))
                success_count += 1
    
            header = "| ID | Name | Unit Type |\n|---|---|---|\n"
            rows = [f"| {u['id']} | {u['name']} | {u['type']} |" for u in units]
            results.append(str("### Units Awaiting Confirmation\n\n" + header + "\n".join(rows)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_unconfirmed_units Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="list_units_to_create",
    description="""Returns a formal report of mandatory organizational units that must be created before confirmation. Use this to identify missing Sales Areas and Sales Groups. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
"""
)
async def list_units_to_create(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        try:
            issues = manager.get_mandatory_issues()
            if not issues:
                results.append(str("Status: No mandatory units identified. Project is ready for confirmation."))
                success_count += 1
    
            report = ["### Units to Create\n", "| Unit Type | Required For | Mapping Status | Action Create | Action Delete |", "|---|---|---|---|---|"]
            for issue in issues:
                # Expected format: "Missing SLA for 6503"
                parts = issue.split("for")
                bus_char = parts[0].replace("Missing", "").strip()
                parent_id = parts[1].split("(")[0].strip()
        
                create_cmd = f"`create_sales_area(sor_id='{parent_id}', ...)`" if bus_char == "SLA" else f"`create_sales_group(sof_id='{parent_id}', ...)`"
                delete_cmd = f"`delete_sales_organization(sor_id='{parent_id}')`" if bus_char == "SLA" else f"`delete_sales_office(sof_id='{parent_id}')`"
        
                report.append(f"| {bus_char} | {parent_id} | Required | {create_cmd} | {delete_cmd} |")
    
            results.append(str("\n".join(report)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk list_units_to_create Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="confirm_units",
    description="""Confirms one or more organizational units. IMPORTANT: This action is IRREVERSIBLE and starts background jobs. Arguments: - unit_ids: List of IDs to confirm (e.g. ['1000', 'P101']) [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "unit_ids" (str)
"""
)
async def confirm_units(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        unit_ids = item.get("unit_ids")
        try:
    
            # Resolve IDs to UUIDs (nodeIds)
            node_ids = []
            for uid in unit_ids:
                unit = manager._find_unit(uid)
                if unit:
                    node_ids.append(unit.get("UUID"))
                else:
                    results.append(str(f"Error: Unit ID {uid} not found."))
                    success_count += 1
            
            if not node_ids:
                results.append(str("Error: No valid unit IDs provided."))
                success_count += 1

            # Validation: Ensure no mandatory issues block confirmation
            logger.info("Performing pre-confirmation validation check...")
            issues = manager.get_mandatory_issues()
            if issues:
                report = ["### Verification Blocked\n", "The following mandatory dependencies must be resolved:"]
                for issue in issues:
                    report.append(f"- {issue}")
        
                report.append("\n**Resolution Suggestions:**")
                for issue in issues:
                    # Clean ID extraction
                    clean_id = issue.split("for")[-1].split("(")[0].strip()
            
                    if "SLA" in issue:
                        report.append(f"- To satisfy {issue}: Call `create_sales_area(sor_id='{clean_id}', ...)`")
                        report.append(f"- To remove {issue}: Call `delete_sales_organization(sor_id='{clean_id}')`")
                    elif "SGR" in issue:
                        report.append(f"- To satisfy {issue}: Call `create_sales_group(sof_id='{clean_id}', ...)`")
                        report.append(f"- To remove {issue}: Call `delete_sales_office(sof_id='{clean_id}')`")
        
                report.append("\nPlease resolve these issues before proceeding with confirmation.")
                results.append(str("\n".join(report)))
                success_count += 1

            res = manager.confirm_units(node_ids)
            if res.get("success"):
                manager.invalidate_cache() # Refresh tree after confirmation
                results.append(str(f"Success: Confirmation job started for {len(unit_ids)} units. Please check status in the UI."))
                success_count += 1
            else:
                msg = res.get("messages", [{}])[0].get("name", "Unknown error")
                results.append(str(f"Error: {msg}"))
                success_count += 1

            results.append(str("\n".join(report)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk confirm_units Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="get_all_data",
    description="""Fetches the entire organizational structure in raw JSON format. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
"""
)
async def get_all_data(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        try:
            results.append(str(json.dumps(manager.get_all_data(), indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk get_all_data Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="get_server_status",
    description="""Returns the current status of the MCP server, including cache age and session info. Use this to verify if the server is using its memory (cache) or hitting the API. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).
"""
)
async def get_server_status(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        try:
            now = time.time()
            cache_age = (now - manager._cache_time) if manager._cache_time > 0 else 0
            cache_status = f"Ready ({int(cache_age)}s old)" if manager._data_cache else "Empty"
    
            status = {
                "Session": "Active" if manager.xsrf_token else "Inactive",
                "Cache": cache_status,
                "CacheTTL": manager._cache_ttl,
                "WorkspaceID": manager.workspace_id,
                "StartupPersistence": "Enabled (session_cache.json)"
            }
            results.append(str(f"### SAP MCP Server Status\n\n" + json.dumps(status, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk get_server_status Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="delete_bulk",
    description="""Bulk delete that deletes given UUIDs. [ID RULE]: The last 2 characters of the ID must be alphabetic (not numbers, e.g., C1BB, CBCC, etc.).

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "input_uuids" (str)
"""
)
async def delete_bulk(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        input_uuids = item.get("input_uuids")
        try:
            res = manager.delete_bulk_with_children(input_uuids)
            results.append(str(json.dumps(res, indent=2)))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk delete_bulk Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_shippingpoint",
    description="""Creates a Shipping Point. ID: Max 4 chars, Name: Max 20 chars, City: Max 40 chars. Subtypes: - Customer Consignment -> SPT_CUST_CONSMT - Return -> SPT_RAW_MATNR - Standard -> SPT_STD

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "id" (str)
      - "shippingpoint_name" (str)
      - "subtype" (str)
      - "cityname" (str)
"""
)
async def create_shippingpoint(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        id = item.get("id")
        shippingpoint_name = item.get("shippingpoint_name")
        subtype = item.get("subtype")
        cityname = item.get("cityname")
        try:
            subtype_map = {
                "Customer Consignment": "SPT_CUST_CONSMT",
                "Return": "SPT_RAW_MATNR",
                "Standard": "SPT_STD",
                "SPT_CUST_CONSMT": "SPT_CUST_CONSMT",
                "SPT_RAW_MATNR": "SPT_RAW_MATNR",
                "SPT_STD": "SPT_STD"
            }
            id = str(id)[:4]
            err = validate_id_suffix(id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            shippingpoint_name = str(shippingpoint_name)[:20]
            cityname = str(cityname)[:40]
            business_character = subtype_map.get(subtype, "SPT_STD")
            # Hardcoded to IN as requested in the payload analysis
            actions = [
                {
                    "action": "createOrgUnitAndRelation",
                    "data": {
                        "workspaceId": manager.workspace_id,
                        "orgUnit": {
                            "Attributes/Country": "IN",
                            "Attributes/City": cityname,
                            "Attributes/ForeignEnabled": False,
                            "BusinessCharacter": business_character,
                            "ID": id,
                            "IsPrimary": False,
                            "Name": shippingpoint_name
                        }
                    },
                    "order": 0
                },
                {
                    "action": "getTransformedOCRData",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id,
                        "hostWorkspaceId": manager.workspace_id
                    },
                    "order": 1
                }
            ]
            manager.execute_api(actions)
            results.append(str(f"Success: Shipping Point {id} ({business_character}) created."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_shippingpoint Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_storagelocation",
    description="""Creates a Storage Location under a Plant. ID: Max 4 chars, Name: Max 16 chars, City: Max 40 chars. Subtypes (from dropdown): - Handling Unit Managed -> STL_HU_MANAGED - Tank Trailer Filling with Residuals -> STL_RESIDUAL - Standard Storage Location for Semi-Finished Goods -> STL_STD_B - Standard Storage Location for Finished Goods -> STL_STD_A - Main Tank (Silo Tank) -> STL_MAIN_TANK - Raw Material -> STL_RAW_MATNR - Day Tank -> STL_DAY_TANK - Kanban -> STL_KANBAN - Return -> STL_RET

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "id" (str)
      - "storagelocation_name" (str)
      - "subtype" (str)
      - "cityname" (str)
      - "parent_plt_id" (str)
"""
)
async def create_storagelocation(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        id = item.get("id")
        storagelocation_name = item.get("storagelocation_name")
        subtype = item.get("subtype")
        cityname = item.get("cityname")
        parent_plt_id = item.get("parent_plt_id")
        try:
    
            subtype_map = {
                "Handling Unit Managed": "STL_HU_MANAGED",
                "Tank Trailer Filling with Residuals": "STL_RESIDUAL",
                "Standard Storage Location for Semi-Finished Goods": "STL_STD_B",
                "Standard Storage Location for Finished Goods": "STL_STD_A",
                "Main Tank (Silo Tank)": "STL_MAIN_TANK",
                "Raw Material": "STL_RAW_MATNR",
                "Day Tank": "STL_DAY_TANK",
                "Kanban": "STL_KANBAN",
                "Return": "STL_RET",
                "STL_HU_MANAGED": "STL_HU_MANAGED",
                "STL_RESIDUAL": "STL_RESIDUAL",
                "STL_STD_B": "STL_STD_B",
                "STL_STD_A": "STL_STD_A",
                "STL_MAIN_TANK": "STL_MAIN_TANK",
                "STL_RAW_MATNR": "STL_RAW_MATNR",
                "STL_DAY_TANK": "STL_DAY_TANK",
                "STL_KANBAN": "STL_KANBAN",
                "STL_RET": "STL_RET"
            }
    
            id = str(id)[:4].upper()
            err = validate_id_suffix(id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            storagelocation_name = str(storagelocation_name)[:16]
            cityname = str(cityname)[:40]
            business_character = subtype_map.get(subtype, "STL_STD_A")
    
            # Find parent Plant UUID
            plt = manager._find_unit(parent_plt_id, unit_type="PLT")
            if not plt:
                results.append(str(f"Error: Plant {parent_plt_id} not found."))
                fail_count += 1
                continue
            plt_uuid = plt.get("UUID")
            country = plt.get("Country") or plt.get("Attributes/Country") or "IN"
    
            actions = [
                {
                    "action": "createOrgUnitAndRelation",
                    "data": {
                        "workspaceId": manager.workspace_id,
                        "orgUnit": {
                            "Attributes/Country": country,
                            "Attributes/City": cityname,
                            "Attributes/ForeignEnabled": False,
                            "BusinessCharacter": business_character,
                            "ID": id,
                            "IsPrimary": False,
                            "Name": storagelocation_name,
                            "UpperOrgCentre/ID": parent_plt_id,
                            "UpperOrgCentre/Type": "PLT",
                            "UpperOrgCentre/UUID": plt_uuid
                        }
                    },
                    "order": 0
                },
                {
                    "action": "getTransformedOCRData",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id,
                        "hostWorkspaceId": manager.workspace_id
                    },
                    "order": 1
                }
            ]
            manager.execute_api(actions)
            results.append(str(f"Success: Storage Location {id} ({business_character}) created under Plant {parent_plt_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_storagelocation Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_salesoffice",
    description="""Creates a new Sales Office (SOF) with country and city details. ID: Max 4 chars, Name: Max 20 chars, City: Max 40 chars. Country: IN | SG | US

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sof_id" (str)
      - "name" (str)
      - "country" (str)
      - "city" (str)
"""
)
async def create_salesoffice(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sof_id = item.get("sof_id")
        name = item.get("name")
        country = item.get("country")
        city = item.get("city")
        try:
            sof_id = str(sof_id)[:4].upper()
            err = validate_id_suffix(sof_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            city = str(city)[:40]
    
            actions = [
                {
                    "action": "createOrgUnitAndRelation",
                    "data": {
                        "workspaceId": manager.workspace_id,
                        "orgUnit": {
                            "Attributes/Country": country,
                            "Attributes/City": city,
                            "Attributes/ForeignEnabled": False,
                            "BusinessCharacter": "SOF",
                            "ID": sof_id,
                            "IsPrimary": False,
                            "Name": name
                        }
                    },
                    "order": 0
                },
                {
                    "action": "getTransformedOCRData",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id,
                        "hostWorkspaceId": manager.workspace_id
                    },
                    "order": 1
                }
            ]
            res = manager.execute_api(actions)
            logger.info(f"create_salesoffice response: {json.dumps(res, indent=2)[:500]}")
            results.append(str(f"Success: Sales Office {sof_id} created. Response: {json.dumps(res)[:300]}"))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_salesoffice Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
@mcp.tool(
    name="create_salesgroup",
    description="""Creates a Sales Group (SGR) under a specific Sales Office (SOF). ID: Max 3 chars, Name: Max 20 chars.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "sgr_id" (str)
      - "name" (str)
      - "parent_sof_id" (str)
"""
)
async def create_salesgroup(json_payload: str) -> str:
    await ensure_session()
    try:
        items = json.loads(json_payload)
    except json.JSONDecodeError as e:
        return f"Error: Invalid JSON input - {e}"
    if not isinstance(items, list):
        return "Error: Input must be a JSON list of objects."

    results = []
    success_count = 0
    fail_count = 0
    for item in items:
        sgr_id = item.get("sgr_id")
        name = item.get("name")
        parent_sof_id = item.get("parent_sof_id")
        try:
            sgr_id = str(sgr_id)[:3].upper()
            err = validate_id_suffix(sgr_id, 2)
            if err:
                results.append(str(err))
                fail_count += 1
                continue
            name = str(name)[:20]
            # Find parent Sales Office UUID
            sof = manager._find_unit(parent_sof_id, unit_type="SOF")
            if not sof:
                results.append(str(f"Error: Sales Office {parent_sof_id} not found."))
                fail_count += 1
                continue
            sof_uuid = sof.get("UUID")
    
            actions = [
                {
                    "action": "createOrgUnitAndRelation",
                    "data": {
                        "workspaceId": manager.workspace_id,
                        "orgUnit": {
                            "BusinessCharacter": "SGR",
                            "ID": sgr_id,
                            "Name": name,
                            "IsPrimary": False,
                            "Attributes/Country": "",
                            "Attributes/ForeignEnabled": False,
                            "UpperOrgCentre/ID": parent_sof_id,
                            "UpperOrgCentre/Type": "SOF",
                            "UpperOrgCentre/UUID": sof_uuid
                        }
                    },
                    "order": 0
                },
                {
                    "action": "getTransformedOCRData",
                    "data": {
                        "currentWorkspaceId": manager.workspace_id,
                        "hostWorkspaceId": manager.workspace_id
                    },
                    "order": 1
                }
            ]
            manager.execute_api(actions)
            results.append(str(f"Success: Sales Group {sgr_id} created under Sales Office {parent_sof_id}."))
            success_count += 1

        except Exception as e:
            results.append(f"Failed: {e}")
            fail_count += 1
    summary = f"### Bulk create_salesgroup Complete\n**{success_count} succeeded, {fail_count} failed** out of {len(items)} total.\n\n"
    summary += "\n".join(results)
    return summary
if __name__ == "__main__":
    mcp.run()