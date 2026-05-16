import asyncio
import requests
import json
from org_structure_tools.crawler import CentralBusinessConfiguration, LOGGER

CBC = CentralBusinessConfiguration()

async def get_workspaceid():
    """Get workspace ID automatically after login - NO CLICKING NEEDED"""
    LOGGER.info("Starting login...")
    login_data = await CBC.login()
    if not login_data:
        LOGGER.error("Login failed!")
        return None
    LOGGER.info("Login successful. Performing dynamic discovery probe...")
    cookies = login_data.get("COOKIES", {})
    xsrf_token = login_data.get("XSRF-TOKEN")
    # Base URL for the API call
    url = CBC.LOGIN_URL.rstrip('/')
    api_url = f"{url}/cbc/project-experience"
    
    # Create a session with the authenticated cookies
    session = requests.Session()
    session.cookies.update(cookies)
    
    # Perfect payload discovered via network analysis in test.py
    # Using "workspace" and "project" root keys instead of "org"
    # These actions do NOT require a workspace ID to be passed in
    payload = {
        "workspace": {"actions": [{"action": "getCurrentWorkspace", "data": {}, "order": 0}]},
        "project": {"actions": [{"action": "getCurrentProject", "data": {}, "order": 1}]}
    }
    headers = {
        "x-csrf-token": xsrf_token, 
        "Content-Type": "application/json",
        "Referer": CBC.LOGIN_URL
    }
    response = session.post(api_url, json=payload, headers=headers)
    if response.status_code != 200:
        LOGGER.error(f"Failed to fetch workspace ID: Status {response.status_code}")
        LOGGER.error(response.text)
        return None
    try:
        resp = response.json()
        # Minimized: Find first 32-char UUID in workspace/data results or headers
        workspace_id = next((v for d in (resp.get("workspace") or resp.get("data") or [])
                            if isinstance(d, dict) for k, v in d.get("data", {}).items()
                            if k.lower() in ["uuid", "workspaceid", "workspaceuuid"] and len(str(v)) == 32), 
                            response.headers.get('x-workspace-id'))
        if workspace_id:
            LOGGER.info(f"✅ Successfully retrieved Workspace ID: {workspace_id}")
            return workspace_id
        LOGGER.warning(f"Workspace ID not found. Response: {resp}")
    except Exception as e:
        LOGGER.error(f"Error parsing discovery response: {e}")
    return None

if __name__ == "__main__":
    asyncio.run(get_workspaceid())