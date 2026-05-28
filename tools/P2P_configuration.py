import os
import asyncio
import logging
import sys
import json
import re
import time
from functools import wraps
from playwright.async_api import async_playwright

HEADLESS = False  # Set to False as requested by user

def time_it(func):
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.perf_counter()
            print(f"⏱️  Function '{func.__name__}' took {end_time - start_time:.2f} seconds to finish.", file=sys.stderr)
    return wrapper

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
EMAIL = "arava@sat-infotech.com"
PASSWORD = "SpringMercury@77"
async def get_webgui_frame(page, timeout_ms=10000):
    start_time = time.time()
    while time.time() - start_time < (timeout_ms / 1000):
        for frame in page.frames:
            if "/sap/bc/gui/sap/its/webgui" in frame.url:
                return frame
        await asyncio.sleep(0.2)
    return None
# LOGIN (FAST + SAFE)
async def login(page, email=None, password=None):
    login_email = email or EMAIL
    login_pwd = password or PASSWORD

    LOGGER.info("WAITING FOR LOGIN PAGE...")

    email_input = page.locator(
        'input[type="email"], input[name="j_username"], input[placeholder="E-Mail"]'
    ).first

    try:
        await email_input.wait_for(state="visible", timeout=30000)
    except:
        LOGGER.info("SESSION ALREADY ACTIVE")
        return

    LOGGER.info(f"LOGIN: {login_email}")
    await email_input.fill(login_email)
    await page.locator(
        'input[type="password"], input[name="j_password"], input[placeholder="Password"]'
    ).first.fill(login_pwd)

    await page.locator(
        'button:has-text("Continue"), button:has-text("Log On"), button:has-text("Sign In")'
    ).first.click()
    
    await asyncio.sleep(1)
# ⚡ FAST SAVE FLOW (NO LOOPS)

async def _dismiss_error_dialog(page):
    """Dismiss any 'Error: Sorry, the app couldn't be opened' UI5 dialog.
    
    Scans ALL frames via JS for error text, then clicks 'Close' button.
    Returns True if a dialog was dismissed, False otherwise.
    """
    JS_CHECK_ERROR = """
    () => {
        const bodyText = (document.body?.innerText || "").toLowerCase();
        if (bodyText.includes("sorry, the app couldn't be opened") ||
            bodyText.includes("sorry, the app could not be opened") ||
            bodyText.includes("app couldn't be opened")) {
            return true;
        }
        // Also check for UI5 error dialog by structure
        const dialogs = document.querySelectorAll('.sapMDialog, [role="dialog"]');
        for (const d of dialogs) {
            const txt = (d.innerText || "").toLowerCase();
            if (txt.includes("error") && txt.includes("close")) return true;
        }
        return false;
    }
    """
    for frame in page.frames:
        try:
            has_error = await frame.evaluate(JS_CHECK_ERROR)
            if has_error:
                LOGGER.warning("ERROR DIALOG DETECTED -- DISMISSING 'APP COULD NOT BE OPENED' POPUP")
                # Try clicking Close button
                for sel in [
                    "button:has-text('Close')",
                    ".sapMBtn:has-text('Close')",
                    "[title='Close']",
                    "button.sapMDialogEndButton",
                    "footer button:has-text('Close')",
                ]:
                    try:
                        btn = frame.locator(sel).first
                        if await btn.count() > 0 and await btn.is_visible(timeout=500):
                            await btn.click(force=True)
                            LOGGER.info("CLICKED 'CLOSE' ON ERROR DIALOG")
                            await asyncio.sleep(0.5)
                            return True
                    except:
                        continue
                # Fallback: try JS click on any button with "Close" text
                try:
                    await frame.evaluate("""
                        () => {
                            const btns = document.querySelectorAll('button, [role="button"]');
                            for (const b of btns) {
                                if ((b.innerText || "").trim().toLowerCase() === "close") {
                                    b.click();
                                    return true;
                                }
                            }
                            return false;
                        }
                    """)
                    LOGGER.info("CLICKED 'CLOSE' ON ERROR DIALOG VIA JS FALLBACK")
                    await asyncio.sleep(0.5)
                    return True
                except:
                    pass
        except:
            pass
    return False

async def execute_save_flow_fast(webgui, page):
    """Wait and click the customizing request 'Continue' (green tick) button."""
    
    # ── Auto-dismiss unexpected "Error" dialogs (e.g. "Sorry, the app couldn't be opened.") ──
    # These appear as UI5 dialogs on the main page, not inside WebGUI frame
    await _dismiss_error_dialog(page)

    await page.keyboard.press("Control+s")
    
    # Try multiple ways to find the 'Continue' / 'OK' / 'Green Tick' button
    selectors = [
        'button:has-text("OK")', 'button:has-text("Continue")',
        'button[title*="Continue"]', 'button[title*="OK"]',
        '[title*="Continue (Enter)"]', '[aria-label*="Continue"]',
        'button[id$="-ok-btn"]', '.sapMBtn:has-text("OK")',
        'button[title*="Checkmark"]'
    ]
    
    start_time = time.time()
    while time.time() - start_time < 5: # Wait up to 5 seconds for dialog
        for sel in selectors:
            try:
                # Search in all frames because the dialog might be in a different frame
                for frame in page.frames:
                    btn = frame.locator(sel).first
                    if await btn.is_visible(timeout=200):
                        await btn.click(force=True)
                        LOGGER.info(f"SAVE CONFIRMED (via {sel})")
                        await asyncio.sleep(1)
                        return True
            except: pass
        await asyncio.sleep(0.2)
        
    LOGGER.warning("NO SAVE POPUP DETECTED OR CLICKED")
    return False

# ⚡ FAST LOCK RELEASE (REDUCED STEPS)
async def release_sap_lock_fast(page):
    try:
        for _ in range(2):
            await page.keyboard.press("F3")
            await asyncio.sleep(0.2) 
    except Exception as e:
        LOGGER.warning(f"LOCK RELEASE ISSUE (IGNORED): {e}")

async def dismiss_lock_dialog(page):
    for frame in page.frames:
        try:
            for btn_text in ["No", "Cancel"]:
                btn = frame.locator(f"button:has-text('{btn_text}')").first
                if await btn.count() > 0 and await btn.is_visible(timeout=500):
                    await btn.click(force=True)
                    await asyncio.sleep(0.2)
                    LOGGER.info(f"LOCK DIALOG DISMISSED via '{btn_text}'.")
                    return
        except:
            pass

async def wait_for_lock_or_ready(page, timeout_ms=4000):
    lock_sel = "text='Locked Data', text='Data locked by', .sapMDialogTitle:has-text('Locked Data')"
    ready_sel = "button:has-text('New Entries'), button:has-text('Deselect All'), .lsTable"
    deadline = asyncio.get_event_loop().time() + timeout_ms / 1000
    while asyncio.get_event_loop().time() < deadline:
        for frame in page.frames:
            try:
                if await frame.locator(lock_sel).count() > 0:
                    return "locked"
                if await frame.locator(ready_sel).count() > 0:
                    return "ready"
            except:
                pass
        await asyncio.sleep(0.15)
    return "timeout"

async def handle_lock_fast(webgui, timeout_ms=1000):
    """Return True if a lock dialog is detected and dismissed (caller should abort)."""
    lock_sel = "text='Locked Data', text='Data locked by', .sapMDialogTitle:has-text('Locked Data')"
    try:
        if await webgui.locator(lock_sel).count() > 0:
            msg = await webgui.locator(lock_sel).first.inner_text()
            LOGGER.error(f"⚠️ LOCK DETECTED: {msg.strip()[:100]}")
            for btn_text in ["No", "Cancel"]:
                btn = webgui.locator(f"button:has-text('{btn_text}')").first
                if await btn.count() > 0:
                    await btn.click(force=True)
                    await asyncio.sleep(0.2)
                    break
            return True
    except:
        pass
    return False

async def install_lock_watcher(page):
    """Attach a Page.on('dialog') handler that auto-dismisses SAP lock dialogs.
    Call this BEFORE page.goto() so the watcher is active from the first request."""
    async def _auto_dismiss(dialog):
        LOGGER.warning(f"AUTO-DISMISSED BROWSER DIALOG: {dialog.message[:80]}")
        await dialog.dismiss()
    page.on("dialog", _auto_dismiss)
    LOGGER.info("🔒 Lock-watcher installed.")

# ───────────────────────────────────────────────────────────────────────────
# MISSING HELPERS ADDED BY AGENT
# ───────────────────────────────────────────────────────────────────────────

async def check_and_abort_if_locked(page):
    """
    Return True (and trigger smart_logout) if SAP shows the 'Locked Data' dialog.

    CRITICAL: The lock dialog often appears AFTER the table/buttons load.
    So when we see 'ready' indicators, we MUST keep scanning for lock
    for a few more seconds before confirming no lock.
    """

    JS_SCAN = """
    () => {
        const LOCK_KEYS = ["locked data", "data is locked", "locked by yourself", "display only"];
        const READY_KEYS = ["new entries", "deselect all"];
        const ERROR_KEYS = ["app couldn't be opened", "app could not be opened"];
        const bodyText = (document.body?.innerText || "").toLowerCase();

        let locked = false;
        let ready = false;
        let error = false;
        for (const k of LOCK_KEYS) {
            if (bodyText.includes(k)) { locked = true; break; }
        }
        for (const k of READY_KEYS) {
            if (bodyText.includes(k)) { ready = true; break; }
        }
        for (const k of ERROR_KEYS) {
            if (bodyText.includes(k)) { error = true; break; }
        }
        if (locked) return "locked";
        if (error) return "error";
        if (ready) return "ready";
        return "unknown";
    }
    """

    ready_seen_at = None  # Track when we FIRST saw ready (to allow lock dialog time to appear)
    EXTRA_WAIT_AFTER_READY = 3.0  # Keep scanning 3 more seconds after seeing ready

    deadline = asyncio.get_event_loop().time() + 20  # 20 second max wait total
    while asyncio.get_event_loop().time() < deadline:
        for frame in page.frames:
            try:
                result = await frame.evaluate(JS_SCAN)

                if result == "locked":
                    LOGGER.error("LOCK DETECTED -- DATA IS LOCKED BY ANOTHER SESSION")
                    # Click "No" to reject displaying locked data
                    for btn_text in ["No", "Cancel", "Close", "OK"]:
                        try:
                            btn = frame.locator(f"button:has-text('{btn_text}')").first
                            if await btn.count() > 0 and await btn.is_visible(timeout=500):
                                await btn.click(force=True)
                                LOGGER.info(f"  CLICKED '{btn_text}' ON LOCK DIALOG")
                                await asyncio.sleep(0.5)
                                break
                        except:
                            pass
                    # Smart logout to close the session cleanly
                    try:
                        await smart_logout(page)
                    except Exception as e:
                        LOGGER.warning(f"  SMART LOGOUT AFTER LOCK FAILED: {e}")
                    return True  # Signal caller to ABORT

                elif result == "error":
                    # "Sorry, the app couldn't be opened" dialog — dismiss and continue
                    LOGGER.warning("ERROR DIALOG DETECTED DURING PAGE LOAD -- DISMISSING...")
                    await _dismiss_error_dialog(page)
                    # Continue scanning after dismissal

                elif result == "ready":
                    if ready_seen_at is None:
                        ready_seen_at = asyncio.get_event_loop().time()
                        LOGGER.info("PAGE LOADED -- WAITING 3s MORE FOR POSSIBLE LOCK DIALOG...")
                    elif asyncio.get_event_loop().time() - ready_seen_at > EXTRA_WAIT_AFTER_READY:
                        # We saw ready 3+ seconds ago and still no lock dialog appeared
                        LOGGER.info("PAGE READY -- NO LOCK DETECTED AFTER EXTRA WAIT")
                        return False

            except Exception:
                pass  # Frame not yet available, keep waiting

        await asyncio.sleep(0.25)

    # If we saw ready but just timed out on the extra wait, that's OK
    if ready_seen_at is not None:
        LOGGER.info("PAGE READY -- NO LOCK DETECTED (TIMEOUT REACHED)")
        return False

    LOGGER.warning("LOCK CHECK TIMEOUT -- PAGE DID NOT SIGNAL READY OR LOCKED. PROCEEDING.")
    return False

async def click_new_entries_button(page):
    """Find and click the 'New Entries' button across all frames.
    
    SAP WebGUI renders toolbar buttons as <div class='lsButton isButton'>,
    NOT as standard <button> elements. We must use WebGUI-specific selectors.
    """
    # Ordered by likelihood — WebGUI-specific selectors first
    selectors = [
        # SAP WebGUI custom button elements (div-based)
        ".lsButton:has-text('New Entries')",
        ".isButton:has-text('New Entries')",
        "[role='button']:has-text('New Entries')",
        "div:has-text('New Entries') >> nth=0",
        # Standard selectors (for Fiori shell toolbar)
        "text='New Entries'",
        "button:has-text('New Entries')",
        "a:has-text('New Entries')",
        "[title*='New Entries']",
        ".sapMBtn:has-text('New Entries')",
        "[aria-label*='New Entries']",
        "[title='New Entries (F5)']",
    ]
    
    # Wait for the page/frames to settle
    await asyncio.sleep(1)
    
    # Try webgui frame first (most likely location), then fall back to all frames
    webgui = await get_webgui_frame(page)
    
    for attempt in range(6):
        # Build frame search order: webgui frame first, then all frames
        frames_to_search = []
        if webgui:
            frames_to_search.append(webgui)
        frames_to_search.extend([f for f in page.frames if f != webgui])
        
        for frame in frames_to_search:
            # Diagnostic: dump what clickable elements exist on attempt 2
            if attempt == 2:
                try:
                    btns = await frame.evaluate("""() => {
                        const els = document.querySelectorAll(
                            'button, a, [role="button"], .lsButton, .isButton, [class*="Btn"]'
                        );
                        return Array.from(els)
                            .map(b => (b.title || b.innerText || '').trim())
                            .filter(t => t.length > 0 && t.length < 80);
                    }""")
                    if btns:
                        LOGGER.info(f"DEBUG [{frame.url[:50]}]: Clickable elements: {btns[:20]}")
                except: pass

            for sel in selectors:
                try:
                    btn = frame.locator(sel).first
                    if await btn.count() > 0:
                        if await btn.is_visible(timeout=500):
                            await btn.click(force=True)
                            LOGGER.info(f"CLICKED 'New Entries' button (via {sel})")
                            await asyncio.sleep(1)
                            return True
                except: continue
        await asyncio.sleep(1)
        
    LOGGER.warning("Could not find 'New Entries' button after all attempts.")
    return False

async def fill_new_inline_row(page, requests):
    """Fill a new row in a table using keyboard navigation."""
    try:
        for idx, req in enumerate(requests):
            val = req.get('want', '')
            if val:
                await page.keyboard.type(str(val), delay=0)
            
            if idx == len(requests) - 1:
                await page.keyboard.press("Enter")
                await asyncio.sleep(1)
            else:
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.15)
        return True
    except Exception as e:
        LOGGER.error(f"Error filling new row: {e}")
        return False

async def execute_save_flow(page):
    """Save changes using common save flow."""
    try:
        webgui = await get_webgui_frame(page)
        return await execute_save_flow_fast(webgui, page)
    except:
        return False

async def graceful_exit(page):
    """Exit the current transaction gracefully."""
    await release_sap_lock_fast(page)
    await asyncio.sleep(1)

async def detect_column_indices(page):
    """Auto-detect column indices based on header row in the grid."""
    # Correct default mapping for 102130 (col 0 = row selection checkbox)
    mapping = {"ShPt": 1, "Plnt": 2, "SC": 3, "Stor": 4}
    try:
        webgui = await get_webgui_frame(page)
        # Read header row (row 0) using the same cell ID pattern as grid scanning
        headers = await webgui.evaluate(r"""
            () => {
                const cols = {};
                const allCells = document.querySelectorAll('[id*="["][id*="]"]');
                for (const cell of allCells) {
                    const m = cell.id.match(/(M0:\d+:\d+)\[0,(\d+)\]/);
                    if (!m) continue;
                    const text = (cell.textContent || '').trim();
                    if (text) cols[text] = parseInt(m[2]);
                }
                return cols;
            }
        """)
        LOGGER.info(f"Auto-detected headers: {headers}")
        for key, idx in headers.items():
            if "ShPt" in key or "Shipping Point" in key: mapping["ShPt"] = idx
            elif "Plnt" in key or "Plant" in key: mapping["Plnt"] = idx
            elif "SC" in key or "Condition" in key: mapping["SC"] = idx
            elif "Stor" in key or "Storage" in key: mapping["Stor"] = idx
    except Exception as e:
        LOGGER.warning(f"Column detection failed, using defaults: {e}")
    return mapping


# LOGOUT
async def smart_logout(page):
    """Closes the SAP session via logoff URL."""
    try:
        LOGGER.info("👋 SIGNING OUT OF SAP SESSION...")
        await page.goto(
            "https://my401292.s4hana.cloud.sap/sap/public/bc/icf/logoff?sap-client=100",
            timeout=10000
        )
        LOGGER.info("✅ SAP SESSION CLOSED.")
    except:
        pass

async def safe_session_cleanup(browser, page):
    """Centralized cleanup to ensure transactional release and browser closure."""
    try:
        LOGGER.info("🔒 RELEASING TRANSACTION & LOGGING OUT...")
        await release_sap_lock_fast(page)
        if not page.is_closed():
            await smart_logout(page)
    except Exception as e:
        LOGGER.debug(f"Cleanup warning: {e}")
    finally:
        try:
            if browser:
                await browser.close()
        except:
            pass

async def get_status_bar_message(frame):
    """
    Extract current status bar text and identify its type (Error, Warning, Success).
    Returns: {"text": str, "type": str} or None
    """
    selectors = [
        "[id*='msgarea']", 
        ".lsMessageBar", 
        ".sapUiMessageBar", 
        "[role='status']",
        ".lsStatusbar__message"
    ]
    
    for sel in selectors:
        try:
            loc = frame.locator(sel).first
            if await loc.is_visible(timeout=500):
                text = (await loc.inner_text()).strip()
                if not text:
                    continue
                
                # Detect type by looking for common SAP status bar icon classes or roles
                m_type = "info"
                html = await loc.inner_html()
                if "Err" in html or "error" in html.lower() or "urMsbBarErr" in html:
                    m_type = "error"
                elif "Warn" in html or "warning" in html.lower() or "urMsbBarWarn" in html:
                    m_type = "warning"
                elif "Succ" in html or "success" in html.lower() or "urMsbBarSucc" in html:
                    m_type = "success"
                
                return {"text": text, "type": m_type}
        except:
            continue
    return None

async def handle_value_help_with_fallback(page, webgui_frame, want):
    """
    PURELY CLICK-BASED SELECTION:
    1. Click the F4 Help button (square-on-square).
    2. Scrutinize the popup for the target value.
    3. Click the target and confirm.
    """
    # Wait for the popup dialog
    LOGGER.info(f"⏳ Waiting for selection popup to find '{want}'...")
    popup_dlg = None
    for _ in range(5):
        for f in page.frames:
            # Flexible match for dialogs
            dlg = f.locator("[role='dialog'], .sapUiWindow, .sapMDialog").filter(has_text=re.compile("Storage Location|Search|Select|Value Help")).first
            if await dlg.is_visible(timeout=500):
                popup_dlg = dlg
                break
        if popup_dlg: break
        await asyncio.sleep(1)
            
    if not popup_dlg:
        LOGGER.error("❌ Selection popup did not appear.")
        return False
            
    # Extract available options using a broader strategy for data cells
    LOGGER.info("📂 Scanning popup for valid entries...")
    
    # 1. Try direct text match first
    target_loc = popup_dlg.locator(f"text='{want}'").first
    match_found = await target_loc.is_visible(timeout=2000)
    
    # 2. Extract and group options for reporting
    options = []
    # SAP WebGUI/Fiori tables often use these for data cells
    cells = popup_dlg.locator("td, [role='gridcell'], span.lsControl--lsStaticText, .sapMText")
    count = await cells.count()
    
    seen = set()
    for i in range(count):
        cell = cells.nth(i)
        text = (await cell.inner_text()).strip()
        if text and text not in seen and len(text) < 60: # Avoid long labels
            seen.add(text)
            options.append(text)
            if not match_found and (text.upper() == want.upper() or text.upper().startswith(want.upper() + " ")):
                target_loc = cell
                match_found = True
    
    if not match_found:
        print("\n" + "="*40, file=sys.stderr)
        print(f"📋 AVAILABLE OPTIONS IN POPUP FOR '{want}':", file=sys.stderr)
        if options:
            for opt in options:
                print(f"   • {opt}", file=sys.stderr)
        else:
            print("   (No data items identified in popup - check frame focus)", file=sys.stderr)
        print("="*40 + "\n", file=sys.stderr)

    if match_found:
         LOGGER.info(f"✅ Found match for '{want}' in list. Clicking...")
         await target_loc.click(force=True)
         await asyncio.sleep(1)
         
         # Click OK in dialog
         ok_selectors = ["button:has-text('OK')", "button[title='OK']", "[id$='-ok-btn']", ".sapMBtn:has-text('OK')"]
         for sel in ok_selectors:
             btn = popup_dlg.locator(sel).first
             if await btn.is_visible(timeout=500):
                 await btn.click(force=True)
                 LOGGER.info("✔️ Clicked OK.")
                 await asyncio.sleep(1)
                 return True
         
         # Fallback to Enter
         await page.keyboard.press("Enter")
         await asyncio.sleep(1)
         return True
    else:
         LOGGER.error(f"❌ '{want}' not found in the clickable list.")
         msg = await get_status_bar_message(webgui_frame)
         if msg: LOGGER.info(f"SAP Status: {msg}")
         
         # Try to close
         await page.keyboard.press("Escape")
         return False



# CORE: lsdata-based row finder and checkbox clicker

# GENERIC LSDATA TOGGLE JS

JS_LSDATA_GENERIC = """
(args) => {
    const { targetText, checkRequests } = args; 
    // checkRequests is an array of objects like: { col: 4, want: true, name: 'OvRel' }
    const results = [];
    
    // Step 1: Find row index and table prefix
    let targetRowIndex = -1;
    let tablePrefix = 'M0:46:1'; // fallback default
    
    const allEls = document.querySelectorAll('[lsdata]');
    for (const el of allEls) {
        try {
            const raw = el.getAttribute('lsdata');
            if (!raw || !raw.includes('"5"')) continue;
            const d = JSON.parse(raw);
            if (d['5'] === targetText) {
                // Typically SID is something like: wnd[0]/usr/tbl.../cmb...-FIELDNAME[1,ROW]
                const sid = (d['21'] && d['21']['SID']) || '';
                const m = sid.match(/\\[(\\d+),(\\d+)\\]/);
                if (m) {
                    targetRowIndex = parseInt(m[2], 10);
                    // Extract table prefix from the element's explicit ID, which looks like M0:46:1[2,1]
                    const idMatch = el.id && el.id.match(/(M0:\\d+:\\d+)\\[/);
                    if (idMatch) tablePrefix = idMatch[1];
                    
                    results.push({ found: true, text: targetText, row: targetRowIndex, tablePrefix: tablePrefix });
                    break;
                }
            }
        } catch(e) {}
    }
    
    // Fallback logic for finding row...
    if (targetRowIndex === -1) {
        const allSpans = document.querySelectorAll('span[id]');
        for (const sp of allSpans) {
            if (sp.textContent.trim() === targetText && sp.id.match(/(M0:\\d+:\\d+)\\[\\d+,\\d+\\]/)) {
                const m = sp.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                if (m) {
                    tablePrefix = m[1];
                    targetRowIndex = parseInt(m[2], 10) - 1; // ID uses 1-based indexing for row
                    results.push({ found: true, text: targetText, rowIndex: targetRowIndex, via: 'id_pattern' });
                    break;
                }
            }
        }
    }
    
    if (targetRowIndex === -1) {
        results.push({ error: 'ROW_NOT_FOUND', targetText: targetText });
        return results;
    }
    
    const sapRow = targetRowIndex + 1; // SAP IDs are 1-based for the row
    
    // Step 2: Toggle each checkbox requested
    for (const request of checkRequests) {
        // e.g., M0:46:1[ROW, COL]_c  (the _c suffix is usually the clickable span inside the cell)
        const cellSpanId = `${tablePrefix}[${sapRow},${request.col}]_c`;
        const cellId = `${tablePrefix}[${sapRow},${request.col}]`;
        
        let el = document.getElementById(cellSpanId) || document.getElementById(cellId);
        
        if (!el) {
            // Backup search by lsdata SIDs for that specific column and row
            const backupCol = `[${request.col},${targetRowIndex}]`;
            for (const sp of allEls) {
                const r = sp.getAttribute('lsdata');
                if (r && r.includes(backupCol) && r.includes('chk')) {
                    el = sp;
                    results.push({ info: `Found by backup SID param: ${backupCol}` });
                    break;
                }
            }
        }

        if (!el) {
            // DEBUG: What columns ARE available for this row?
            const availableCols = [];
            const prefixMatches = document.querySelectorAll(`[id^="${tablePrefix}[${sapRow},"]`);
            for (const sp of prefixMatches) {
                if (sp.id.endsWith('_c') || sp.tagName === 'INPUT') {
                    availableCols.push(sp.id);
                }
            }
            results.push({ error: `${request.name}_NOT_FOUND`, spanId: cellSpanId, debug_available_checkboxes: availableCols });
            continue;
        }
        
        try {
            const raw = el.getAttribute('lsdata');
            let isChecked = false;
            if (raw) {
                const d = JSON.parse(raw);
                isChecked = d['1'] === true; // '1' is the semantic 'checked' state in SAP UI
            } else if (el.tagName === 'INPUT' && el.type === 'checkbox') {
                isChecked = el.checked;
            }
            
            results.push({ step: `${request.name}_state`, isChecked, want: request.want, needsToggle: isChecked !== request.want });
            
            if (isChecked === request.want) {
                results.push({ step: `${request.name}_already_correct` });
                continue;
            }
            
            el.click();
            results.push({ step: `${request.name}_clicked` });
        } catch(e) {
            results.push({ error: `${request.name}_error: ${e.message}` });
        }
    }5
    
    return results;
}
"""

async def find_and_toggle_checkboxes_generic(page, target_text, check_requests):
    """
    Universally finds specific row by text, then hits specified checkboxes by column index.
    check_requests should be a list like: [{'col': 4, 'want': True, 'name': 'OvRel'}]
    """
    webgui_frame = await get_webgui_frame(page)
    if not webgui_frame:
        LOGGER.error("WEBGUI IFRAME NOT FOUND!")
        return False

    LOGGER.info(f"WEBGUI FRAME FOUND: {webgui_frame.url[:80]}")

    max_scrolls = 15
    for scroll_attempt in range(max_scrolls):
        LOGGER.info(f"--- SCROLL ATTEMPT {scroll_attempt + 1}/{max_scrolls} ---")

        # Execute generic search and toggle
        try:
            result = await webgui_frame.evaluate(
                JS_LSDATA_GENERIC, 
                {"targetText": target_text, "checkRequests": check_requests}
            )
            LOGGER.info(f"[GENERIC MATCH] Result: {json.dumps(result, indent=2)}")
            
            has_error = any(isinstance(r, dict) and 'error' in r and 'NOT_FOUND' in str(r.get('error', '')) for r in result)
            has_click = any(isinstance(r, dict) and r.get('step', '').endswith('_clicked') for r in result)
            has_correct = any(isinstance(r, dict) and r.get('step', '').endswith('_already_correct') for r in result)
            
            if has_click or has_correct:
                LOGGER.info("[GENERIC MATCH] SUCCESS!")
                await asyncio.sleep(1)
                return True
                
        except Exception as e:
            LOGGER.error(f"[GENERIC MATCH] JS Error: {e}")

        # Strategy 3: Playwright fallback removed because it's no longer necessary with generic approach 
        # and JS can directly manipulate table IDs universally.
            
        # --- Not Found in Current View. Try Scrolling ---
        LOGGER.info(f"'{target_text}' not found in current view. Pressing PageDown to fetch more rows...")
        try:
            # Click inside the table body or any span in the table to gain focus before pressing PageDown
            focus_target = webgui_frame.locator("table[id*='M0:46:1'] tbody, span[lsdata]").first
            if await focus_target.is_visible(timeout=2000):
                await focus_target.click(force=True)
            else:
                await webgui_frame.locator("body").click(force=True)
                
            await page.keyboard.press("PageDown")
            # SAP WebGUI needs a moment to request and render the next chunk of rows
            await asyncio.sleep(1)
        except Exception as e:
            LOGGER.error(f"Failed to scroll: {e}")

    LOGGER.error(f"ALL STRATEGIES FAILED: '{target_text}' NOT FOUND AFTER {max_scrolls} SCROLLS.")
    return False


# SPECIALIZED PO RESTART HELPER (To avoid breaking generic logic)

JS_LSDATA_PO_RESTART = """
(args) => {
    const { targetText, checkRequests } = args; 
    const results = [];
    const targetTextLower = targetText.toLowerCase();
    
    let targetRowIndex = -1;
    let tablePrefix = 'M0:46:1'; 
    
    // Step 1: Find the row by attribute text via lsdata
    const allEls = document.querySelectorAll('[lsdata]');
    for (const el of allEls) {
        try {
            const raw = el.getAttribute('lsdata');
            if (!raw) continue;
            const d = JSON.parse(raw);
            if (d['5'] && d['5'].toLowerCase() === targetTextLower) {
                const sid = (d['21'] && d['21']['SID']) || '';
                const m = sid.match(/\\[(\\d+),(\\d+)\\]/);
                if (m) {
                    targetRowIndex = parseInt(m[2], 10);
                    const idMatch = el.id && el.id.match(/(M0:\\d+:\\d+)\\[/);
                    if (idMatch) tablePrefix = idMatch[1];
                    results.push({ found: true, text: d['5'], row: targetRowIndex, tablePrefix: tablePrefix });
                    break;
                }
            }
        } catch(e) {}
    }
    
    if (targetRowIndex === -1) {
        return [{ error: 'ROW_NOT_FOUND', targetText: targetText }];
    }
    
    const sapRow = targetRowIndex + 1;
    
    // Step 2: Process each column request
    for (const request of checkRequests) {
        const cellId = `${tablePrefix}[${sapRow},${request.col}]`;
        let cell = document.getElementById(cellId);
        
        if (!cell) {
            results.push({ error: `${request.name}_NOT_FOUND`, col: request.col });
            continue;
        }
        
        try {
            if (request.type === 'checkbox') {
                // Find the actual checkbox input inside the cell
                let chk = cell.querySelector('input[type="checkbox"]');
                if (!chk) chk = cell.querySelector('[role="checkbox"]');
                
                let isChecked = false;
                if (chk) {
                    isChecked = chk.checked || chk.getAttribute('aria-checked') === 'true';
                } else {
                    const raw = cell.getAttribute('lsdata');
                    if (raw) { const d = JSON.parse(raw); isChecked = d['1'] === true; }
                }
                
                if (isChecked === request.want) {
                    results.push({ step: `${request.name}_already_correct`, checked: isChecked });
                } else {
                    if (chk) chk.click(); else cell.click();
                    results.push({ step: `${request.name}_clicked`, was: isChecked, now: request.want });
                }
            } else if (request.type === 'combobox') {
                // Find the INPUT textbox inside the cell (e.g. input#M0:46:1[9,3]_c)
                let inp = cell.querySelector('input');
                if (!inp) inp = document.getElementById(cellId + '_c');
                const inputId = inp ? inp.id : null;
                
                // DON'T click anything here - let Python handle the interaction
                results.push({ 
                    step: `${request.name}_found`, 
                    type: 'combobox', 
                    want: request.want, 
                    inputId: inputId, 
                    cellId: cellId 
                });
            }
        } catch(e) {
            results.push({ error: `${request.name}_error: ${e.message}` });
        }
    }
    return results;
}
"""

async def find_and_set_po_restart_conditions(page, target_text, requests):
    """Specialized helper for PO Restart Conditions that handles checkboxes and comboboxes."""
    webgui_frame = await get_webgui_frame(page)
    if not webgui_frame: return False

    for scroll_attempt in range(5):
        try:
            result = await webgui_frame.evaluate(JS_LSDATA_PO_RESTART, {"targetText": target_text, "checkRequests": requests})
            LOGGER.info(f"[PO RESTART] Result: {json.dumps(result, indent=2)}")
            
            if any('error' in r for r in result):
                await webgui_frame.locator("body").click(force=True)
                await page.keyboard.press("PageDown")
                await asyncio.sleep(1)
                continue
            
            # Handle combobox: wait for enabled, click, type, and verify
            got_all = True
            for r in result:
                if r.get('type') == 'combobox' and r.get('inputId'):
                    target_val = r.get('want')
                    input_id = r.get('inputId')
                    LOGGER.info(f"SETTING '{target_val}' in input #{input_id}...")
                    
                    try:
                        # 1. Wait for element to be enabled and visible
                        inp = webgui_frame.locator(f"id={input_id}")
                        await inp.wait_for(state="visible", timeout=10000)
                        
                        # 2. Focus and open
                        await inp.click(timeout=5000)
                        await asyncio.sleep(0.2)
                        
                        # 3. Enter value via keyboard
                        await page.keyboard.press("Control+a")
                        await page.keyboard.press("Backspace")
                        await page.keyboard.type(target_val, delay=0)
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(0.8)
                        
                        # 4. Verify selection
                        current_text = await inp.inner_text()
                        if not current_text:
                             # Try getting value attribute for native inputs
                             current_text = await inp.get_attribute("value") or ""
                             
                        if target_val.lower() not in current_text.lower():
                            LOGGER.warning(f"Verification failed! Expected '{target_val}', got '{current_text}'. Retrying with Alt+Down fallback...")
                            await page.keyboard.press("Alt+ArrowDown")
                            await asyncio.sleep(0.2)
                            await page.keyboard.type(target_val[0], delay=0)
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(0.2)
                        
                        await page.keyboard.press("Tab")
                        LOGGER.info(f"✔️ Finished setting '{target_val}'")
                    except Exception as e:
                        LOGGER.error(f"Failed to set combobox {input_id}: {e}")
                        got_all = False
            return got_all
        except Exception as e:
            LOGGER.error(f"Error: {e}")
            await asyncio.sleep(1)
    return False


async def find_and_set_material_groups(page, target_text, requests):
    """
    Column mapping: Matl Group (1), Desc (2), AGrp (3), DUW (4), Desc2 (5)
    """
    webgui_frame = await get_webgui_frame(page)
    if not webgui_frame:
        return False

    # We no longer force a jump/scroll right away. We try finding it in the current view first!

    for scroll_attempt in range(12):
        try:
            row_info = await webgui_frame.evaluate("""
                (targetText) => {
                    const targetTextLower = targetText.toLowerCase();
                    
                    // Strategy 1: Search by text content and traverse up to find row ID
                    const allVisible = document.querySelectorAll('span, div, input, td');
                    for (const el of allVisible) {
                        if (el.textContent.trim().toLowerCase() === targetTextLower) {
                            let curr = el;
                            while (curr && curr !== document.body) {
                                if (curr.id && curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/)) {
                                    const m = curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                                    return { prefix: m[1], rowIdx: m[2], id: curr.id };
                                }
                                curr = curr.parentElement;
                            }
                        }
                    }

                    // Strategy 2: Fallback to lsdata iteration
                    const lsElements = document.querySelectorAll('[lsdata]');
                    for (const el of lsElements) {
                        const lsDataStr = el.getAttribute('lsdata');
                        try {
                            const d = JSON.parse(lsDataStr.replace(/'/g, '"'));
                            for (let k in d) {
                                if (d[k] && typeof d[k] === 'string' && d[k].toLowerCase() === targetTextLower) {
                                    let curr = el;
                                    while (curr && curr !== document.body) {
                                        if (curr.id && curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/)) {
                                            const m = curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                                            return { prefix: m[1], rowIdx: m[2], id: curr.id };
                                        }
                                        curr = curr.parentElement;
                                    }
                                }
                            }
                        } catch(e) { continue; }
                    }
                    return null;
                }
            """, target_text)

            if not row_info:
                # If not found immediately, we can use 'Position...' to jump if it's the first failed attempt
                if scroll_attempt == 0:
                    try:
                        pos_btn = webgui_frame.locator("button:has-text('Position...'), [title*='Position'], .sapMBtn:has-text('Position')").first
                        if await pos_btn.is_visible(timeout=500):
                            await pos_btn.click(force=True)
                            await asyncio.sleep(0.2)
                            await page.keyboard.type(target_text, delay=0)
                            await page.keyboard.press("Enter")
                            LOGGER.info(f"Target '{target_text}' not immediately visible. Used 'Position...' to jump.")
                            await asyncio.sleep(1)
                            continue
                    except: pass
                
                # Otherwise, just scroll down blindly
                await webgui_frame.locator("body").click(force=True)
                await page.keyboard.press("PageDown")
                await asyncio.sleep(0.2)
                continue

            prefix = row_info['prefix']
            row_idx = row_info['rowIdx']

            # --------------------------------------------------------
            # MANDATORY LEFT FOCUS: Click ID (Col 1) to force view left
            # --------------------------------------------------------
            id_col_id = f"{prefix}[{row_idx},1]_c"
            id_locator = webgui_frame.locator(f"id={id_col_id}")
            if await id_locator.count() > 0:
                await id_locator.first.click(force=True)
                await asyncio.sleep(1) # Wait for horizontal scroll
            # --------------------------------------------------------

            all_ok = True
            for idx, req in enumerate(requests):
                col = req['col']
                want = req['want']
                full_id = f"{prefix}[{row_idx},{col}]_c"
                input_locator = webgui_frame.locator(f"id={full_id}")
                
                try:
                    if await input_locator.count() > 0:
                        # Center the element horizontally and vertically to ensure it's in the middle of the viewport
                        await webgui_frame.evaluate("""(id) => {
                            const el = document.getElementById(id);
                            if (!el) return;
                            el.scrollIntoView({ behavior: 'instant', block: 'center', inline: 'center' });
                            // Force focus via JS if it's tricky
                            const inp = (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') ? el : el.querySelector('input, textarea');
                            if (inp) { inp.focus(); }
                        }""", full_id)
                        await asyncio.sleep(0.2)

                        # 1. Click to focus (using fallback if standard click fails)
                        try:
                            await input_locator.first.click(force=True, timeout=1000)
                        except:
                            LOGGER.warning(f"Standard click failed for '{req['name']}'. Using JS fallback.")
                            await webgui_frame.evaluate("(id) => { document.getElementById(id)?.dispatchEvent(new MouseEvent('mousedown', {bubbles: true})); }", full_id)
                        
                        await asyncio.sleep(0.2)
                        
                        # 2. Press F2 to ensure the cell is in "Edit Mode"
                        await page.keyboard.press("F2")
                        await asyncio.sleep(0.2)
                        
                        # 2.5 FORCE FOCUS & CLEAR:
                        await webgui_frame.evaluate("""(id) => {
                            let cell = document.getElementById(id);
                            if (!cell) return;
                            let inp = (cell.tagName === 'INPUT' || cell.tagName === 'TEXTAREA') ? cell : cell.querySelector('input, textarea');
                            if (inp) {
                                inp.focus();
                                inp.value = '';
                                inp.dispatchEvent(new Event('input', { bubbles: true }));
                            }
                        }""", full_id)
                        await asyncio.sleep(0.3)
                        
                        # 4. Type new value (SAP registers this as a user physical edit because we type from an empty stat)
                        if want:
                            await page.keyboard.type(str(want), delay=0)
                            await asyncio.sleep(0.2)
                        
                        # 5. Commit logic:
                        # For the first field (usually Description), we press Enter to lock it in.
                        # For the last field, we press Enter to finish the row.
                        # For intermediate fields, we press Tab.
                        if idx == 0 or idx == len(requests) - 1:
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(1) # Increased for stability
                        else:
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.3)

                        LOGGER.info(f"SET '{req['name']}' to '{want}'")
                    else:
                        LOGGER.error(f"INPUT NOT FOUND FOR '{req['name']}' (Row {row_idx}, Col {col})")
                        all_ok = False
                except Exception as e:
                    LOGGER.error(f"FAILED TO SET '{req['name']}': {e}")
                    all_ok = False
            
            return all_ok

        except Exception as e:
            LOGGER.error(f"Error in find_and_set_material_groups loop: {e}")
            await asyncio.sleep(0.2)
            
    return False

async def handle_sap_confirmation_dialogs(page, timeout_ms=500):
    """Common logic to find and click 'Continue', 'OK', or the green checkmark in SAP dialogs."""
    dialog_found = False
    for frame in page.frames:
        try:
            tick = frame.locator(
                "button[title*='Continue'], button[title*='OK'], "
                "[title*='Continue (Enter)'], [aria-label*='Continue'], "
                "button:has-text('Continue'), button:has-text('OK'), "
                "button[title*='Checkmark'], .sapMBtn:has-text('OK')"
            ).first
            if await tick.is_visible(timeout=timeout_ms):
                await tick.click(force=True)
                LOGGER.info("CONFIRMED SAP DIALOG (Tick/Continue).")
                await asyncio.sleep(1)
                dialog_found = True
        except: pass
    return dialog_found


# SAVE FLOW

async def select_from_sap_value_help(page, value):
    """
    Handles SAP F4 Value Help dialog. Opens, searches for value, and selects it.
    """
    LOGGER.info(f"OPENING VALUE HELP for '{value}'...")
    # 1. Trigger F4
    await page.keyboard.press("F4")
    await asyncio.sleep(1)
    
    # 2. Look for the dialog in all frames
    dialog_frame = None
    for frame in page.frames:
        if await frame.locator(".sapMDialog, [role='dialog']").count() > 0:
            dialog_frame = frame
            break
            
    if not dialog_frame:
        LOGGER.error("Value Help dialog not found!")
        return False
        
    try:
        # 3. Search for the value in the dialog's search field
        # Try 'Find expression' (Fiori) or standard search box
        search_input = dialog_frame.locator("input[placeholder*='Find'], input[placeholder*='Search'], [id*='searchField'] input").first
        if await search_input.is_visible(timeout=2000):
            await search_input.click()
            await page.keyboard.press("Control+a")
            await page.keyboard.press("Backspace")
            await page.keyboard.type(str(value), delay=0)
            await page.keyboard.press("Enter")
            
            # Explicitly click the 'Go' button if present to refresh the list
            go_btn = dialog_frame.locator("button:has-text('Go'), [id*='search'] button:has-text('Go')").first
            if await go_btn.is_visible(timeout=1000):
                await go_btn.click()
                await asyncio.sleep(1)
        
        # 4. Find and select the value in the results table
        LOGGER.info(f"Searching for item '{value}' in Value Help table...")
        
        # We'll try a direct JavaScript click as a more robust fallback
        clicked = await dialog_frame.evaluate("""
            (targetVal) => {
                // Find all potential row/cell elements in the dialog
                const elements = document.querySelectorAll('.sapMDialog [role="gridcell"], .sapMDialog td, .sapMDialog span, .sapMDialog .sapMLIB');
                for (const el of elements) {
                    if (el.innerText && el.innerText.trim() === targetVal) {
                        el.scrollIntoView({ block: 'center' });
                        const rect = el.getBoundingClientRect();
                        const event = new MouseEvent('click', {
                            view: window,
                            bubbles: true,
                            cancelable: true,
                            clientX: rect.left + rect.width / 2,
                            clientY: rect.top + rect.height / 2
                        });
                        el.dispatchEvent(event);
                        return true;
                    }
                }
                return false;
            }
        """, str(value))
        
        if not clicked:
            # Try a second JS pass with a broader search but exact text match
            clicked = await dialog_frame.evaluate("""
                (targetVal) => {
                    const allNodes = document.evaluate(
                        `.//*[text()='${targetVal}']`, 
                        document, null, XPathResult.ANY_TYPE, null
                    );
                    let node = allNodes.iterateNext();
                    if (node) {
                        node.scrollIntoView({ block: 'center' });
                        node.click();
                        return true;
                    }
                    return false;
                }
            """, str(value))

        if clicked:
            LOGGER.info(f"Target item '{value}' clicked via JS.")
            await asyncio.sleep(1)
            # Confirm selection with OK button or Enter
            ok_btn = dialog_frame.locator("button:has-text('OK'), button:has-text('Select'), button[id*='ok']").first
            if await ok_btn.is_visible(timeout=1000):
                await ok_btn.click()
            else:
                await page.keyboard.press("Enter")
                
            LOGGER.info(f"SUCCESS: Set value to '{value}' via Value Help.")
            await asyncio.sleep(1)
            return True
        else:
            LOGGER.error(f"Value '{value}' not found in results list even with JS evaluation.")
            return False
            
    except Exception as e:
        LOGGER.error(f"Error in Value Help selection: {e}")
        return False


async def find_and_set_multi_key_row(page, keys, requests):
    """
    Finds a row in an SAP grid matching multiple key columns and updates target columns.
    keys: list of {'col': int, 'val': str}
    requests: list of {'col': int, 'want': str, 'name': str}
    """
    webgui_frame = await get_webgui_frame(page)
    if not webgui_frame:
        return False

    # 1. Try Position... button approach first (most reliable for navigation)
    try:
        pos_btn = webgui_frame.locator(".lsButton:has-text('Position'), button:has-text('Position'), [title*='Position']").first
        if await pos_btn.is_visible(timeout=1000):
            await pos_btn.click(force=True)
            await asyncio.sleep(1)
            # Position dialog fields are usually in order of keys
            for key in keys:
                await page.keyboard.type(str(key['val']), delay=0)
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.3)
            await page.keyboard.press("Enter")
            LOGGER.info(f"Used 'Position...' for: {[k['val'] for k in keys]}")
            await asyncio.sleep(3)
    except Exception as e:
        LOGGER.info(f"Position button logic skipped: {e}")

    for scroll_attempt in range(20):
        try:
            # 2. Scanning Strategy: Use DOM IDs directly and match content with fuzzy number support
            row_info = await webgui_frame.evaluate("""
                (keys) => {
                    const rowMatches = {}; // rowID -> count of matched keys
                    const allCells = document.querySelectorAll('[id*="["][id*="]"]');
                    
                    for (const cell of allCells) {
                        const m = cell.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                        if (!m) continue;
                        
                        const prefix = m[1], rowIdx = m[2], colIdx = parseInt(m[3]);
                        const key = keys.find(k => k.col === colIdx);
                        
                        if (key) {
                            // Try multiple ways to get the value (span text or input value)
                            const input = cell.querySelector('input');
                            const rawVal = (input ? input.value : cell.textContent).trim().toLowerCase();
                            const targetVal = key.val.toString().toLowerCase();
                            
                            // Fuzzy match (allow leading zeros mismatch if both are valid numbers)
                            const isMatch = (rawVal === targetVal) || 
                                           (parseInt(rawVal) === parseInt(targetVal) && rawVal !== "" && targetVal !== "");
                            
                            if (isMatch) {
                                const rowKey = prefix + "_" + rowIdx;
                                if (!rowMatches[rowKey]) rowMatches[rowKey] = { prefix, rowIdx, matches: new Set() };
                                rowMatches[rowKey].matches.add(key.col);
                            }
                        }
                    }
                    
                    for (const k in rowMatches) {
                        if (rowMatches[k].matches.size === keys.length) return rowMatches[k];
                    }
                    return null;
                }
            """, keys)

            if not row_info:
                LOGGER.info(f"Row {keys} not found on current page (Attempt {scroll_attempt+1}/20). Scrolling...")
                await webgui_frame.locator("body").click(force=True)
                await page.keyboard.press("PageDown")
                await asyncio.sleep(2.5)
                continue

            prefix = row_info['prefix']
            row_idx = row_info['rowIdx']
            LOGGER.info(f"MATCH FOUND at Prefix={prefix}, RowIdx={row_idx}")

            all_ok = True
            for idx, req in enumerate(requests):
                col = req['col']
                want = req['want']
                full_id = f"{prefix}[{row_idx},{col}]_c"
                input_locator = webgui_frame.locator(f"id={full_id}")
                
                try:
                    if await input_locator.count() > 0:
                        await input_locator.first.scroll_into_view_if_needed()
                        
                        # Optimization: check if we even need to type
                        if want != "" and want is not None:
                            # Safely get current value from input or text
                            try:
                                current_val = await input_locator.first.locator("input").input_value() if await input_locator.first.locator("input").count() > 0 else await input_locator.first.inner_text()
                                current_val = current_val.strip()
                                if current_val == str(want).strip():
                                    LOGGER.info(f"STATUS: Skipped '{req['name']}' (already '{want}')")
                                    continue
                            except Exception as e:
                                pass # proceed to type if check fails
                                
                        await input_locator.first.click(force=True)
                        await asyncio.sleep(0.2)
                        await page.keyboard.press("F2")
                        await asyncio.sleep(0.8)
                        if req.get('help_fallback'):
                            # PURE CLICK APPROACH:
                            # 1. Click the cell
                            await input_locator.first.click(force=True)
                            await asyncio.sleep(0.2)
                            
                            # 2. Click the F4 icon (square-on-square)
                            LOGGER.info("🔍 Clicking F4 Help icon...")
                            vh_icon = input_locator.locator("span[id*='helpbutton'], [title*='Help'], .lsField-inputfieldhelpbutton").first
                            if await vh_icon.is_visible(timeout=1500):
                                await vh_icon.click(force=True)
                            else:
                                await page.keyboard.press("F4")
                            
                            # 3. Handle selection in popup
                            h_success = await handle_value_help_with_fallback(page, webgui_frame, want)
                            if not h_success: all_ok = False
                        elif req.get('type') == 'value_help':
                            success = await select_from_sap_value_help(page, want)
                            if not success: all_ok = False
                        else:
                            # Standard typing logic
                            await page.keyboard.press("Control+a")
                            await page.keyboard.press("Backspace")
                            for _ in range(6):
                                await page.keyboard.press("Backspace")
                                await page.keyboard.press("Delete")
                            await asyncio.sleep(0.2)
                            
                            if want:
                                await page.keyboard.type(str(want), delay=0)
                                await asyncio.sleep(0.2)
                            
                            if idx == len(requests) - 1:
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(1)
                            else:
                                await page.keyboard.press("Tab")
                                await asyncio.sleep(0.2)

                        LOGGER.info(f"STATUS: Processed '{req['name']}'")
                    else:
                        LOGGER.error(f"CELL NOT FOUND: {full_id} for '{req['name']}'")
                        all_ok = False
                except Exception as e:
                    LOGGER.error(f"FAILED TO SET '{req['name']}': {e}")
                    all_ok = False
            
            return all_ok

        except Exception as e:
            LOGGER.error(f"Error in multi-key loop: {e}")
            await asyncio.sleep(1)
            
    return False


# MAIN FLOWS

@time_it
async def activate_purchase_requisition_workflow_102911(targets: list[dict]):
    """Main flow for 'Purchase Organization Configuration'."""
    LOGGER.info("🚀 STARTING: Purchase Requisition Workflow (SSCUI 102911)")
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SLOG_PR_FLEX_WF&CustomizingObject=VV_T161_VF_PRWFL&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_52001450&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"🌐 NAVIGATING TO SAP URL: {url[:80]}...")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            
            for target in targets:
                target_name = target.get("target_name")
                opt1_val = target.get("opt1_val")
                opt2_val = target.get("opt2_val")
                
                LOGGER.info(f"PROCESSING: {target_name}")
                requests = [
                    {'col': 3, 'want': opt1_val, 'name': 'OvRel'},
                    {'col': 4, 'want': opt2_val, 'name': 'Sce'}
                ]
                success = await find_and_toggle_checkboxes_generic(page, target_name, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY TOGGLED CHECKBOXES FOR '{target_name}'!")
                else:
                    LOGGER.error(f"FAILED TO TOGGLE CHECKBOXES FOR '{target_name}'.")

            # Execute save flow and check for success
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

@time_it
async def configure_purchase_requisition_102888(
    targets: list[dict]
):
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=MMPUR_V_WFLRSTRT&CustomizingObject=MMPUR_V_WFLRSTRT&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_52001363&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)

            # Local clean JS toggle implementation specifically for this function
            JS_TOGGLE = """
            (args) => {
                const { targetText, checkRequests } = args; 
                const results = [];
                let targetRowIndex = -1;
                let tablePrefix = 'M0:46:1'; 
                
                const allEls = document.querySelectorAll('[lsdata]');
                for (const el of allEls) {
                    try {
                        const raw = el.getAttribute('lsdata');
                        if (!raw || !raw.includes('"5"')) continue;
                        const d = JSON.parse(raw);
                        if (d['5'] === targetText) {
                            const sid = (d['21'] && d['21']['SID']) || '';
                            const m = sid.match(/\\[(\\d+),(\\d+)\\]/);
                            if (m) {
                                targetRowIndex = parseInt(m[2], 10);
                                const idMatch = el.id && el.id.match(/(M0:\\d+:\\d+)\\[/);
                                if (idMatch) tablePrefix = idMatch[1];
                                results.push({ found: true, text: targetText, row: targetRowIndex, tablePrefix: tablePrefix });
                                break;
                            }
                        }
                    } catch(e) {}
                }
                
                if (targetRowIndex === -1) {
                    const allSpans = document.querySelectorAll('span[id]');
                    for (const sp of allSpans) {
                        if (sp.textContent.trim() === targetText && sp.id.match(/(M0:\\d+:\\d+)\\[\\d+,\\d+\\]/)) {
                            const m = sp.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                            if (m) {
                                tablePrefix = m[1];
                                targetRowIndex = parseInt(m[2], 10) - 1;
                                results.push({ found: true, text: targetText, rowIndex: targetRowIndex, via: 'id_pattern' });
                                break;
                            }
                        }
                    }
                }
                
                if (targetRowIndex === -1) {
                    results.push({ error: 'ROW_NOT_FOUND', targetText: targetText });
                    return results;
                }
                
                const sapRow = targetRowIndex + 1;
                
                for (const request of checkRequests) {
                    const cellSpanId = `${tablePrefix}[${sapRow},${request.col}]_c`;
                    const cellId = `${tablePrefix}[${sapRow},${request.col}]`;
                    let el = document.getElementById(cellSpanId) || document.getElementById(cellId);
                    
                    if (!el) {
                        const backupCol = `[${request.col},${targetRowIndex}]`;
                        for (const sp of allEls) {
                            const r = sp.getAttribute('lsdata');
                            if (r && r.includes(backupCol) && r.includes('chk')) {
                                el = sp;
                                results.push({ info: `Found by backup SID param: ${backupCol}` });
                                break;
                            }
                        }
                    }
            
                    if (!el) {
                        results.push({ error: `${request.name}_NOT_FOUND`, spanId: cellSpanId });
                        continue;
                    }
                    
                    try {
                        const raw = el.getAttribute('lsdata');
                        let isChecked = false;
                        if (raw) {
                            const d = JSON.parse(raw);
                            isChecked = d['1'] === true;
                        } else if (el.tagName === 'INPUT' && el.type === 'checkbox') {
                            isChecked = el.checked;
                        }
                        
                        results.push({ step: `${request.name}_state`, isChecked, want: request.want, needsToggle: isChecked !== request.want });
                        if (isChecked === request.want) {
                            results.push({ step: `${request.name}_already_correct` });
                            continue;
                        }
                        el.click();
                        results.push({ step: `${request.name}_clicked` });
                    } catch(e) {
                        results.push({ error: `${request.name}_error: ${e.message}` });
                    }
                }
                return results;
            }
            """

            async def find_and_toggle_scrolling(target_name, check_requests):
                # 1. Scroll all the way to the top first
                LOGGER.info(f"Scrolling to the top of the table for '{target_name}'...")
                try:
                    focus_target = webgui.locator("table[id*='M0:46:1'] tbody, span[lsdata]").first
                    if await focus_target.is_visible(timeout=2000):
                        await focus_target.click(force=True)
                    else:
                        await webgui.locator("body").click(force=True)
                except Exception as e:
                    LOGGER.warning(f"Failed to focus for scrolling up: {e}")
                
                for _ in range(15):
                    await page.keyboard.press("PageUp")
                    await asyncio.sleep(0.1)
                await asyncio.sleep(0.5)

                # 2. Search Top to Bottom
                LOGGER.info(f"Searching top to bottom for '{target_name}'...")
                max_scrolls = 15
                for scroll_attempt in range(max_scrolls):
                    LOGGER.info(f"Top-to-Bottom Scroll Attempt {scroll_attempt + 1}/{max_scrolls}")
                    try:
                        result = await webgui.evaluate(
                            JS_TOGGLE, 
                            {"targetText": target_name, "checkRequests": check_requests}
                        )
                        LOGGER.info(f"Evaluation result: {json.dumps(result)}")
                        has_error = any(isinstance(r, dict) and 'error' in r and 'ROW_NOT_FOUND' in str(r.get('error', '')) for r in result)
                        has_click = any(isinstance(r, dict) and r.get('step', '').endswith('_clicked') for r in result)
                        has_correct = any(isinstance(r, dict) and r.get('step', '').endswith('_already_correct') for r in result)
                        
                        if not has_error and (has_click or has_correct):
                            LOGGER.info(f"✅ Found and processed '{target_name}' during top-to-bottom search!")
                            await asyncio.sleep(1)
                            return True
                    except Exception as e:
                        LOGGER.error(f"JS Error: {e}")

                    # Scroll down
                    try:
                        focus_target = webgui.locator("table[id*='M0:46:1'] tbody, span[lsdata]").first
                        if await focus_target.is_visible(timeout=2000):
                            await focus_target.click(force=True)
                        else:
                            await webgui.locator("body").click(force=True)
                        await page.keyboard.press("PageDown")
                        await asyncio.sleep(1.2)
                    except Exception as e:
                        LOGGER.error(f"Failed to scroll down: {e}")

                # 3. Search Bottom to Top
                LOGGER.info(f"'{target_name}' not found top-to-bottom. Searching bottom-to-top...")
                for scroll_attempt in range(max_scrolls):
                    LOGGER.info(f"Bottom-to-Top Scroll Attempt {scroll_attempt + 1}/{max_scrolls}")
                    try:
                        result = await webgui.evaluate(
                            JS_TOGGLE, 
                            {"targetText": target_name, "checkRequests": check_requests}
                        )
                        LOGGER.info(f"Evaluation result: {json.dumps(result)}")
                        has_error = any(isinstance(r, dict) and 'error' in r and 'ROW_NOT_FOUND' in str(r.get('error', '')) for r in result)
                        has_click = any(isinstance(r, dict) and r.get('step', '').endswith('_clicked') for r in result)
                        has_correct = any(isinstance(r, dict) and r.get('step', '').endswith('_already_correct') for r in result)
                        
                        if not has_error and (has_click or has_correct):
                            LOGGER.info(f"✅ Found and processed '{target_name}' during bottom-to-top search!")
                            await asyncio.sleep(1)
                            return True
                    except Exception as e:
                        LOGGER.error(f"JS Error: {e}")

                    # Scroll up
                    try:
                        focus_target = webgui.locator("table[id*='M0:46:1'] tbody, span[lsdata]").first
                        if await focus_target.is_visible(timeout=2000):
                            await focus_target.click(force=True)
                        else:
                            await webgui.locator("body").click(force=True)
                        await page.keyboard.press("PageUp")
                        await asyncio.sleep(1.2)
                    except Exception as e:
                        LOGGER.error(f"Failed to scroll up: {e}")

                LOGGER.error(f"❌ Failed to find target '{target_name}' after full scroll up and down.")
                return False

            for idx, target in enumerate(targets, start=1):
                target_name = target.get("target_name")
                ss_proc = target.get("ss_proc")
                prf_proc = target.get("prf_proc")

                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING: {target_name}")

                requests = [
                    {'col': 3, 'want': ss_proc, 'name': 'SS Proc'},
                    {'col': 4, 'want': prf_proc, 'name': 'Prf Proc'}
                ]

                try:
                    success = await find_and_toggle_scrolling(target_name, requests)
                    if success:
                        LOGGER.info(f"SUCCESS: {target_name}")
                    else:
                        LOGGER.error(f"FAILED: {target_name}")
                except Exception as e:
                    LOGGER.error(f"ERROR processing {target_name}: {e}")
                    if "SAP_LOCK_DETECTED" in str(e):
                        LOGGER.warning("🔒 LOCK DETECTED — skipping remaining items.")
                        break
                await asyncio.sleep(0.3)
            
            LOGGER.info("EXECUTING SAVE FLOW...")
            await asyncio.sleep(1)
            saved = await execute_save_flow_fast(webgui, page)
            LOGGER.info("ALL OPERATIONS COMPLETED SUCCESSFULLY")

        finally:
            await safe_session_cleanup(browser, page)

@time_it
async def activate_po_flexible_workflow_101097(targets: list[dict]):
    """Main flow for 'Document Types Purchase order' configuration."""
    LOGGER.info("🚀 STARTING: Activate PO Flexible Workflow (SSCUI 101097)")
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SLOG_PO_FLEX_WF&CustomizingObject=VV_T161_VF_WFL&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_52000654&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"🌐 NAVIGATING TO SAP URL: {url[:80]}...")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page)
            if await check_and_abort_if_locked(page):
                return
            
            for target in targets:
                doc_type = target.get("Type")
                scenario_based = target.get("Scenario_based_workflow")
                
                LOGGER.info(f"PROCESSING: {doc_type}")
                requests = [
                    {'col': 3, 'want': scenario_based, 'name': 'Scenario Based Workflow'}
                ]
                success = await find_and_toggle_checkboxes_generic(page, doc_type, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY TOGGLED CHECKBOXES FOR '{doc_type}'!")
                else:
                    LOGGER.error(f"FAILED TO TOGGLE CHECKBOXES FOR '{doc_type}'.")

            # Execute save flow and check for success
            webgui = await get_webgui_frame(page)
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)


@time_it
async def configure_po_workflow_restart_conditions_103345(targets: list[dict]):
    """Main flow for 'Manage Conditions to Restart PO Flex. WF'."""
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=MMPUR_V_PO_WFL_R&CustomizingObject=MMPUR_V_PO_WFL_R&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_52001716&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            
            for idx, target in enumerate(targets, start=1):
                attr = target.get("Purchase_Order_Attributes")
                enable = target.get("Enable")
                # Handle case sensitivity for 'restart_type'
                restart_type = target.get("Restart_Type") or target.get("restart_type")
                
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING ATTRIBUTE: {attr}")
                # Col 2: Enable (Checkbox)
                # Col 3: Restart Type (Combobox)
                requests = [
                    {'col': 2, 'want': enable, 'type': 'checkbox', 'name': 'Enable'}
                ]
                if restart_type:
                    requests.append({'col': 3, 'want': restart_type, 'type': 'combobox', 'name': 'Restart Type'})
                
                success = await find_and_set_po_restart_conditions(page, attr, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY SET: {attr}")
                else:
                    LOGGER.error(f"FAILED TO SET: {attr}")
                
                # Brief pause to let SAP UI settle between operations
                await asyncio.sleep(1)
                
            # Execute save flow and check for success
            saved = await execute_save_flow_fast(webgui, page)
            
            # CAPTURE FINAL STATE SCREENSHOT
            screenshot_path = os.path.join(os.path.dirname(__file__), "final_state_po_restart.png")
            await page.screenshot(path=screenshot_path) 
            LOGGER.info(f"CAPTURED FINAL STATE SCREENSHOT: {screenshot_path}")
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)



@time_it
async def Create_New_Material_Group_102665(targets: list[dict]):
    """
    Main flow to CREATE NEW Material Group entries.
    Clicks 'New Entries' once and fills all rows sequentially.
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLMSOMSF&CustomizingObject=V023&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87002257&Type=SSCUI"
    
    def format_id(val):
        return str(val).strip().upper() if val else ""
    def format_title_case(val):
        if not val: return ""
        return " ".join([word.capitalize() for word in str(val).strip().split()])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        LOGGER.info(f"OPENING CREATE FLOW: {url}")
        await install_lock_watcher(page)
        await page.goto(url, wait_until="commit", timeout=60000)
        try:
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                await browser.close()
                return
        except Exception as e:
            LOGGER.error(f"Login failed: {e}")
            await browser.close()
            return

        await asyncio.sleep(1)
        
        # Step 1: Click 'New Entries' once to enter the bulk entry screen
        LOGGER.info("NAVIGATING TO 'NEW ENTRIES: ADDED ENTRIES' SCREEN...")
        if not await click_new_entries_button(page):
            LOGGER.error("COULD NOT START NEW ENTRIES FLOW.")
            await browser.close()
            return
            
        await asyncio.sleep(1)

        # Step 2: Loop through targets and fill rows sequentially
        for idx, target in enumerate(targets, start=1):
            matl_group = format_id(target.get("Matl_Group"))
            desc = format_title_case(target.get("Material_Group_Desc"))
            agrp = format_id(target.get("AGrp")) 
            duw = format_id(target.get("DUW"))   
            desc2 = format_title_case(target.get("Description_2"))
            
            LOGGER.info(f"[{idx}/{len(targets)}] FILLING DATA FOR: {matl_group}")
            
            requests = [
                {'col': 1, 'want': matl_group, 'name': 'Matl Group'},
                {'col': 2, 'want': desc,       'name': 'Description'},
                {'col': 3, 'want': agrp,       'name': 'AGrp'},
                {'col': 4, 'want': duw,        'name': 'DUW'},
                {'col': 5, 'want': desc2,      'name': 'Description 2'}
            ]

            # fill_new_inline_row uses Tab/Enter to move through cells and rows.
            # Focus is automatically on the next row after the previous Enter.
            success = await fill_new_inline_row(page, requests)
            if success:
                LOGGER.info(f"ROW {idx} ({matl_group}) PREPARED SUCCESSFULLY.")
            else:
                LOGGER.error(f"FAILED TO FILL ROW {idx} ({matl_group}).")
        
        LOGGER.info("ALL ROWS FILLED. EXECUTING SAVE...")
        await execute_save_flow(page)
        
        LOGGER.info("INITIATING GRACEFUL EXIT...")
        await graceful_exit(page)
        await browser.close()

@time_it
async def Edit_Existing_Material_Group_102665(targets: list[dict]):
    """
    Main flow to EDIT EXISTING Material Group entries.
    Handles login, session setup, and row lookup loop.
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLMSOMSF&CustomizingObject=V023&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87002257&Type=SSCUI"
    
    def format_id(val):
        return str(val).strip().upper() if val else ""
    def format_title_case(val):
        if not val: return ""
        return " ".join([word.capitalize() for word in str(val).strip().split()])

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        LOGGER.info(f"OPENING EDIT FLOW: {url}")
        await install_lock_watcher(page)
        await page.goto(url, wait_until="commit", timeout=60000)
        try:
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                await browser.close()
                return
        except Exception as e:
            LOGGER.error(f"Login failed: {e}")
            await browser.close()
            return

        await asyncio.sleep(1)
        for idx, target in enumerate(targets, start=1):
            matl_group = format_id(target.get("Matl_Group"))
            desc = format_title_case(target.get("Material_Group_Desc"))
            agrp = format_id(target.get("AGrp")) 
            duw = format_id(target.get("DUW"))   
            desc2 = format_title_case(target.get("Description_2"))
            
            LOGGER.info(f"[{idx}/{len(targets)}] EDITING EXISTING ENTRY: {matl_group}")
            
            requests = []
            if desc:  requests.append({'col': 2, 'want': desc,  'name': 'Description'})
            if agrp:  requests.append({'col': 3, 'want': agrp,  'name': 'AGrp'})
            if duw:   requests.append({'col': 4, 'want': duw,   'name': 'DUW'})
            if desc2: requests.append({'col': 5, 'want': desc2, 'name': 'Description 2'})

            success = await find_and_set_material_groups(page, matl_group, requests)
            if success:
                LOGGER.info(f"SUCCESSFULLY UPDATED: {matl_group}")
            else:
                LOGGER.warning(f"ID '{matl_group}' NOT FOUND IN TABLE.")
        
        await execute_save_flow(page)
        await graceful_exit(page)
        await browser.close()





async def search_via_position_dialog_102130(page, shpt, plnt, sc):
    """Searches for a picking location row using the Position dialog."""
    webgui = await get_webgui_frame(page)
    if not webgui:
        return False
    try:
        # SAP WebGUI renders Position as a div.lsButton, NOT a <button>
        pos_btn = None
        # Primary: div.lsButton with text 'Position'
        try:
            pos_btn = webgui.locator(".lsButton:has-text('Position')").first
            await pos_btn.wait_for(state="visible", timeout=2000)
            LOGGER.info("Position button found via .lsButton:has-text('Position')")
        except Exception as e1:
            LOGGER.info(f"Selector .lsButton:has-text('Position') failed: {e1}")
            # Fallback: Playwright text selector
            try:
                pos_btn = webgui.locator("text=Position...").first
                await pos_btn.wait_for(state="visible", timeout=2000)
                LOGGER.info("Position button found via text=Position...")
            except Exception as e2:
                LOGGER.info(f"Selector text=Position... failed: {e2}")
                # Fallback: any element with title containing 'Position'
                try:
                    pos_btn = webgui.locator("[title*='Position']").first
                    await pos_btn.wait_for(state="visible", timeout=2000)
                    LOGGER.info("Position button found via [title*='Position']")
                except Exception as e3:
                    LOGGER.info(f"Selector [title*='Position'] failed: {e3}")
                    pos_btn = None
        if not pos_btn:
            LOGGER.error("Position dialog button not found via any selector")
            return False

        await pos_btn.click(force=True)
        await asyncio.sleep(1.5)
        
        # Position dialog fields: ShPt, Plnt, SC
        # Select all and clear before typing
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(str(shpt), delay=0)
        await page.keyboard.press("Tab")
        await asyncio.sleep(0.3)
        
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(str(plnt), delay=0)
        await page.keyboard.press("Tab")
        await asyncio.sleep(0.3)
        
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Backspace")
        await page.keyboard.type(str(sc), delay=0)
        await page.keyboard.press("Enter")
        
        LOGGER.info(f"Submitted Position dialog for: {shpt}/{plnt}/{sc}")
        await asyncio.sleep(3)
        return True
    except Exception as e:
        LOGGER.error(f"Error in search_via_position_dialog: {e}")
    return False


async def find_matching_row_index_102130(page, shpt, plnt, sc, col_map):
    """Checks the visible grid to find a row matching the specified ShPt, Plnt, and SC."""
    webgui = await get_webgui_frame(page)
    if not webgui:
        return None
    try:
        # First, dump all visible cell data for debugging
        all_cells_data = await webgui.evaluate(r"""
            () => {
                const result = [];
                const allCells = document.querySelectorAll('[id*="["][id*="]"]');
                for (const cell of allCells) {
                    const m = cell.id.match(/(M0:\d+:\d+)\[(\d+),(\d+)\]/);
                    if (!m) continue;
                    const input = cell.querySelector('input');
                    const val = (input ? input.value : cell.textContent).trim();
                    if (val) {
                        result.push({
                            prefix: m[1],
                            row: m[2],
                            col: parseInt(m[3]),
                            val: val,
                            id: cell.id
                        });
                    }
                }
                return result;
            }
        """)

        # Log what we found to help debug
        if all_cells_data:
            # Group by row for cleaner logging
            rows_data = {}
            for cell in all_cells_data:
                row_key = f"{cell['prefix']}_{cell['row']}"
                if row_key not in rows_data:
                    rows_data[row_key] = {}
                rows_data[row_key][cell['col']] = cell['val']
            
            LOGGER.info(f"🔍 GRID SCAN: Found {len(rows_data)} rows in DOM. Looking for ShPt={shpt}, Plnt={plnt}, SC={sc}")
            for rk, cols in list(rows_data.items())[:5]:  # Log first 5 rows
                LOGGER.info(f"   Row {rk}: {cols}")

        # Now do the actual matching
        row_info = await webgui.evaluate(r"""
            (args) => {
                const keys = args.keys;
                const allCells = document.querySelectorAll('[id*="["][id*="]"]');
                const rowMatches = {};
                
                for (const cell of allCells) {
                    const m = cell.id.match(/(M0:\d+:\d+)\[(\d+),(\d+)\]/);
                    if (!m) continue;
                    
                    const prefix = m[1], rowIdx = m[2], colIdx = parseInt(m[3]);
                    const key = keys.find(k => k.col === colIdx);
                    
                    if (key) {
                        const input = cell.querySelector('input');
                        const rawVal = (input ? input.value : cell.textContent).trim();
                        const targetVal = key.val.toString().trim();
                        
                        // Fuzzy match: exact string OR numeric equivalence
                        const isMatch = (rawVal.toLowerCase() === targetVal.toLowerCase()) || 
                                       (rawVal !== "" && targetVal !== "" && 
                                        !isNaN(rawVal) && !isNaN(targetVal) && 
                                        parseInt(rawVal) === parseInt(targetVal));
                        
                        if (isMatch) {
                            const rowKey = prefix + "_" + rowIdx;
                            if (!rowMatches[rowKey]) rowMatches[rowKey] = { prefix, rowIdx, matches: new Set(), matchedCols: [] };
                            rowMatches[rowKey].matches.add(key.col);
                            rowMatches[rowKey].matchedCols.push({col: colIdx, raw: rawVal, target: targetVal});
                        }
                    }
                }
                
                // Find row where ALL keys matched
                for (const k in rowMatches) {
                    if (rowMatches[k].matches.size === keys.length) {
                        return {
                            prefix: rowMatches[k].prefix,
                            rowIdx: rowMatches[k].rowIdx,
                            matchedCols: rowMatches[k].matchedCols
                        };
                    }
                }
                
                // Return partial matches for debugging
                const partials = [];
                for (const k in rowMatches) {
                    partials.push({
                        row: k,
                        matchCount: rowMatches[k].matches.size,
                        matchedCols: rowMatches[k].matchedCols
                    });
                }
                return partials.length > 0 ? { partial: true, matches: partials } : null;
            }
        """, {
            "keys": [
                {"col": col_map["ShPt"], "val": shpt},
                {"col": col_map["Plnt"], "val": plnt},
                {"col": col_map["SC"], "val": sc}
            ]
        })

        if row_info and not row_info.get("partial"):
            LOGGER.info(f"✅ MATCH FOUND: prefix={row_info['prefix']}, rowIdx={row_info['rowIdx']}, cols={row_info.get('matchedCols')}")
            return row_info
        elif row_info and row_info.get("partial"):
            LOGGER.warning(f"⚠️ PARTIAL MATCHES ONLY (need 3 keys, got): {row_info['matches']}")
            return None
        else:
            LOGGER.info(f"❌ NO MATCH for ShPt={shpt}, Plnt={plnt}, SC={sc} with col_map={col_map}")
            return None
    except Exception as e:
        LOGGER.error(f"Error finding row index: {e}")
    return None


@time_it
async def Create_New_Picking_Locations_102130(targets: list[dict]):
    """
    Unified flow to CREATE or EDIT Picking Location assignments (SSCUI 102130).
    Processes targets one-by-one: searches first; if found, edits Storage Location and saves;
    if not found, clicks 'New Entries', fills details, saves, and returns to main screen.
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLSDOVL3&CustomizingObject=V_TVKOL&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87006701&Type=SSCUI"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        LOGGER.info(f"OPENING FLOW: {url}")
        await install_lock_watcher(page)
        await page.goto(url, wait_until="commit", timeout=60000)
        try:
            await login(page)
            if await check_and_abort_if_locked(page):
                await browser.close()
                return
        except Exception as e:
            LOGGER.error(f"Login failed: {e}")
            await browser.close()
            return

        # Auto-detect actual DOM column indices from the grid header
        col_map = await detect_column_indices(page)
        LOGGER.info(f"Using column map: {col_map}")

        for idx, target in enumerate(targets, start=1):
            shpt = str(target.get("ShPt", "")).upper().strip()
            plnt = str(target.get("Plnt", "")).upper().strip()
            sc   = str(target.get("SC", "")).upper().strip()
            stor = str(target.get("Stor", "")).upper().strip()

            LOGGER.info(f"[{idx}/{len(targets)}] Processing target: {shpt}/{plnt}/{sc}")

            # 1. Search via Position dialog
            success = await search_via_position_dialog_102130(page, shpt, plnt, sc)
            if not success:
                LOGGER.error(f"Could not use Position dialog for {shpt}/{plnt}/{sc}. Skipping target.")
                continue

            # 2. Check if the row exists on the screen
            row_info = await find_matching_row_index_102130(page, shpt, plnt, sc, col_map)
            if row_info:
                LOGGER.info(f"Row {shpt}/{plnt}/{sc} EXISTS (RowIdx={row_info['rowIdx']}). Editing Storage Location...")
                prefix = row_info['prefix']
                row_idx = row_info['rowIdx']
                col = col_map["Stor"]
                full_id = f"{prefix}[{row_idx},{col}]_c"
                
                webgui = await get_webgui_frame(page)
                input_locator = webgui.locator(f"id={full_id}")
                
                try:
                    if await input_locator.count() > 0:
                        await input_locator.first.scroll_into_view_if_needed()
                        
                        # Optimization: check if already matching
                        try:
                            current_val = await input_locator.first.locator("input").input_value() if await input_locator.first.locator("input").count() > 0 else await input_locator.first.inner_text()
                            if current_val.strip() == str(stor).strip():
                                LOGGER.info(f"Skipping edit (already set to '{stor}')")
                                continue
                        except:
                            pass
                        
                        await input_locator.first.click(force=True)
                        await asyncio.sleep(0.2)
                        await page.keyboard.press("F2")
                        await asyncio.sleep(0.8)
                        
                        # Standard typing/clearing logic
                        await page.keyboard.press("Control+a")
                        await page.keyboard.press("Backspace")
                        for _ in range(6):
                            await page.keyboard.press("Backspace")
                            await page.keyboard.press("Delete")
                        await asyncio.sleep(0.2)
                        
                        if stor:
                            await page.keyboard.type(str(stor), delay=0)
                            await asyncio.sleep(0.2)
                        
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(1)
                        LOGGER.info(f"Successfully edited Row: {shpt}/{plnt}/{sc}")
                        
                        # Save inline change
                        await execute_save_flow(page)
                        await asyncio.sleep(1)
                except Exception as e:
                    LOGGER.error(f"Failed to edit Row {shpt}/{plnt}/{sc}: {e}")
            else:
                LOGGER.info(f"Row {shpt}/{plnt}/{sc} DOES NOT exist. Creating new entry...")
                if not await click_new_entries_button(page):
                    LOGGER.error("Could not click 'New Entries' button.")
                else:
                    await asyncio.sleep(1.5)
                    
                    new_entry_fields = [
                        {'col': col_map["ShPt"], 'want': shpt, 'name': 'Shipping Point'},
                        {'col': col_map["Plnt"], 'want': plnt, 'name': 'Plant'},
                        {'col': col_map["SC"],   'want': sc,   'name': 'Shipping Condition'},
                        {'col': col_map["Stor"], 'want': stor, 'name': 'Storage Loc', 'help_fallback': True},
                    ]
                    success = await fill_new_inline_row(page, new_entry_fields)
                    if success:
                        LOGGER.info(f"ROW ({shpt}/{plnt}/{sc}) PREPARED.")
                        
                        # Save new entry
                        await execute_save_flow(page)
                        await asyncio.sleep(1)
                        
                        # Capture SAP status bar message after save
                        webgui = await get_webgui_frame(page)
                        if webgui:
                            status = await get_status_bar_message(webgui)
                            if status:
                                if status["type"] == "error":
                                    LOGGER.error(f"❌ SAP ERROR after save: {status['text']}")
                                    # If entry already exists, cancel the new entry screen
                                    if "already exists" in status["text"].lower() or "same key" in status["text"].lower():
                                        LOGGER.warning(f"Entry {shpt}/{plnt}/{sc} already exists. Cancelling new entry...")
                                        await page.keyboard.press("F12")  # Cancel
                                        await asyncio.sleep(1)
                                        # Handle "Data will be lost" confirmation if it appears
                                        try:
                                            for f in page.frames:
                                                yes_btn = f.locator(".lsButton:has-text('Yes'), button:has-text('Yes')").first
                                                if await yes_btn.is_visible(timeout=500):
                                                    await yes_btn.click(force=True)
                                                    await asyncio.sleep(1)
                                                    break
                                        except:
                                            pass
                                elif status["type"] == "warning":
                                    LOGGER.warning(f"⚠️ SAP WARNING after save: {status['text']}")
                                elif status["type"] == "success":
                                    LOGGER.info(f"✅ SAP SUCCESS: {status['text']}")
                                else:
                                    LOGGER.info(f"ℹ️ SAP STATUS: {status['text']}")
                    else:
                        LOGGER.error(f"FAILED TO FILL ROW ({shpt}/{plnt}/{sc}).")
                    
                    # Return to main list view (F3)
                    LOGGER.info("Returning to main list view...")
                    await page.keyboard.press("F3")
                    await asyncio.sleep(2.5)

        await graceful_exit(page)
        await browser.close()


@time_it
async def Edit_Existing_Picking_Locations_102130(targets: list[dict]):
    """Edit existing picking locations by calling the unified Create_New_Picking_Locations_102130."""
    await Create_New_Picking_Locations_102130(targets)


@time_it
async def supply_invoices_101098(payment_block: bool, release_completed: bool, check_auth: bool):
    """Main flow for 'Activate Flexible Workflow for Supplier Invoices' (SSCUI 101098)."""
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=T169WF02&CustomizingObject=T169WF02&CustomizingObjectType=S&CustomizingProject=&CustomizingTransaction=S_ER9_52000674&Type=SSCUI"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI FRAME NOT FOUND. EXITING...")
                return

            # Toggle the checkboxes in SAP based on parameters
            try:
                # Find all checkboxes in the list
                checkboxes = webgui_frame.locator("[role='checkbox']")
                count = await checkboxes.count()
                
                if count < 3:
                    LOGGER.error(f"Found only {count} checkboxes, expected at least 3.")
                    return
                
                vals = [payment_block, release_completed, check_auth]
                names = [
                    "Payment Block: Flexible Workflow is Active",
                    "Release Completed Invoice: Flexible Workflow is Active",
                    "Check Authorizations for Flexible Workflow Steps"
                ]

                for i, want_checked in enumerate(vals):
                    cb = checkboxes.nth(i)
                    is_checked = await cb.get_attribute("aria-checked") == "true"
                    if is_checked != want_checked:
                        await cb.click(force=True)
                        LOGGER.info(f"Set '{names[i]}' to {want_checked}")
                        await asyncio.sleep(0.2)
                    else:
                        LOGGER.info(f"'{names[i]}' already set to {want_checked}")
            except Exception as e:
                LOGGER.error(f"Failed to toggle checkboxes: {e}")

            # Execute save flow and check for success
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

@time_it
async def Document_Types_Contract_Change_101247(targets: list[dict]):
    """Main flow for 'Document Types Contract Change' (SSCUI 101247)."""
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SLOG_CTR_FLEX_WF&CustomizingObject=VV_T161_VK_WFL&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_52000771&Type=SSCUI"
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            
            for target in targets:
                type_code = target.get("type_code")
                scenario_based = target.get("Scenario_based_workflow")
                
                LOGGER.info(f"PROCESSING: {type_code}")
                requests = [
                    {'col': 3, 'want': scenario_based, 'name': 'Scenario Based Workflow'}
                ]
                
                success = await find_and_toggle_checkboxes_generic(page, type_code, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY TOGGLED CHECKBOXES FOR '{type_code}'!")
                else:
                    LOGGER.error(f"FAILED TO TOGGLE CHECKBOXES FOR '{type_code}'.")
                
                await asyncio.sleep(1)
                
            # Execute save flow
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)



@time_it
async def Entry_Aids_for_Items_Without_a_Material_Master_101602(targets: list[dict]):
    """Main flow for 'Entry Aids for Items Without a Material Master' (SSCUI 101602)."""
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLMEOMQW&CustomizingObject=V023_E&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87002214&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            
            for idx, target in enumerate(targets, start=1):
                mat_grp = target.get("Mat_Grp")
                descr = target.get("Mat_Grp_Descr")
                valcl = target.get("ValCl")
                purvalk = target.get("PurValK")
                
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING MAT GRP: {mat_grp}")
                
                requests = []
                if descr is not None: requests.append({'col': 2, 'want': descr, 'name': 'Mat. Grp Descr.'})
                if valcl is not None: requests.append({'col': 3, 'want': valcl, 'name': 'ValCl'})
                if purvalk is not None: requests.append({'col': 4, 'want': purvalk, 'name': 'PurValK'})
                
                # Reusing the robust material groups helper as the table structure is identical
                success = await find_and_set_material_groups(page, mat_grp, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY UPDATED: {mat_grp}")
                else:
                    LOGGER.error(f"FAILED TO UPDATE: {mat_grp}")
                
                await asyncio.sleep(1)
                
            # Execute save flow
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

@time_it
async def maintain_purchasing_organization_105939(targets: list[dict]):
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUSAPCOX08&CustomizingObject=V_T024E&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87007552&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page)
            if await check_and_abort_if_locked(page):
                return
            
            for idx, target in enumerate(targets, start=1):
                porg = target.get("POrg")
                desc = target.get("Description")
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING PORG: {porg}")
                requests = []
                if desc: 
                    desc = desc[:20]
                    requests.append({'col': 2, 'want': desc, 'name': 'Description'})
                success = await find_and_set_material_groups(page, porg, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY UPDATED: {porg}")
                else:
                    LOGGER.error(f"FAILED TO UPDATE: {porg}")
            webgui = await get_webgui_frame(page)
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

@time_it
async def Assign_Shipping_Points_102126(targets: list[dict]):
    """Main flow for 'Assign Shipping Points' (SSCUI 102126)."""
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CMMENUOLSDOVL2&CustomizingObject=V_TVSTZ&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87006845&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
                if await check_and_abort_if_locked(page):
                    return
            except Exception as e:
                LOGGER.error(f"Login failed: {e}")
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)
            
            # Diagnostic Log to inspect columns
            try:
                diag_info = await webgui.evaluate("""
                    () => {
                        const info = [];
                        const cells = Array.from(document.querySelectorAll('[id]')).filter(el => el.id.includes('[0,'));
                        cells.forEach(c => {
                            const m = c.id.match(/\\[(\\\\d+),(\\\\d+)\\]/);
                            if (m) {
                                const input = c.querySelector('input');
                                info.push({
                                    col: parseInt(m[2]),
                                    val: (input ? input.value : c.textContent).trim(),
                                    tooltip: c.title || c.getAttribute('title') || ''
                                });
                            }
                        });
                        return info;
                    }
                """)
                LOGGER.info(f"DIAGNOSTIC COLUMNS FOR ROW 0: {diag_info}")
            except Exception as ex:
                LOGGER.error(f"Failed to get diagnostic columns: {ex}")
            
            for idx, target in enumerate(targets, start=1):
                sc = target.get("Shipping_Condition")
                lg = target.get("Loading_Group")
                pl = target.get("Plant")
                prop = target.get("Proposed_Shipping_Point")
                alts = target.get("Alternative_Shipping_Points", [])
                
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING Shipping Pt Determination: SC={sc}, LG={lg}, PL={pl}")
                
                keys = [
                    {'col': 1, 'val': sc},
                    {'col': 2, 'val': lg},
                    {'col': 3, 'val': pl}
                ]
                requests = []
                if prop is not None:
                    requests.append({'col': 4, 'want': prop, 'name': 'Proposed Shipping Point'})
                for i, alt_val in enumerate(alts):
                    requests.append({'col': 5 + i, 'want': alt_val, 'name': f'Alt. ShipPt {i+1}'})
                
                success = await find_and_set_multi_key_row(page, keys, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY UPDATED Row: {sc}/{lg}/{pl}")
                else:
                    LOGGER.error(f"FAILED TO FIND OR UPDATE Row: {sc}/{lg}/{pl}")
                
                await asyncio.sleep(1)
                
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

async def edit_storage_location_105933(storageid, city, description):
    import sys, os, requests
    sys.path.append(os.path.join(os.path.dirname(__file__), "org_structure_tools"))
    from org_structure_tools.alltools import (
        update_organizational_unit, get_headers, ensure_session, 
        CBC_BASE_URL, WORKSPACEID, LOGGER
    )
    if not city:
        LOGGER.info(f"CITY NOT PROVIDED. FETCHING CBC JSON FOR '{storageid}'...")
        await ensure_session()
        payload = {"org": {"actions": [{"action": "getOrgUnitsTillLevel", "data": {"currentWorkspaceId": WORKSPACEID, "hostWorkspaceId": WORKSPACEID, "level": 10}, "order": 0}]}}
        raw_data = requests.post(CBC_BASE_URL, json=payload, headers=get_headers(), timeout=60).json()
        units = raw_data.get("data", [{}])[0].get("data", [])
        id_map = {str(u.get("Id")).upper(): u for u in units if u.get("Id")}
        
        def find_city_recursive(uid, visited=None):
            if visited is None: visited = set()
            uid = str(uid).upper()
            if not uid or uid in visited: return ""
            visited.add(uid)
            unit = id_map.get(uid)
            if not unit: return ""
            city_val = unit.get("Attributes", {}).get("City")
            if city_val: return city_val
            parent_id = unit.get("ParentID")
            return find_city_recursive(parent_id, visited) if parent_id else ""
        city = find_city_recursive(storageid)
    name = description[:16] if description else ""
    LOGGER.info(f"UPDATING STORAGE LOCATION '{storageid}' -> NAME: '{name}', CITY: '{city}'")
    response = await update_organizational_unit(unit_id=storageid, name=name, city=city)
    print(f"API RESPONSE FOR {storageid}: {response}", file=sys.stderr)
    return response


@time_it
async def create_purchase_groups_102914(targets: list[dict]):
    """
    Main flow for 'Create Purchasing Groups' (SSCUI 102914).
     Column mapping (based on UI5 table layout):
     - Pur. Grp: Col 1 (Identifier)
     - Desc. Pur. Grp: Col 2 (Limit: 18 chars)
     - Tel.No. Pur.Grp: Col 3 (Limit: 12 chars)
     - Fax Number: Col 4 (Limit: 31 chars)
     - Telephone: Col 5 (Limit: 30 chars)
     - Extension: Col 6 (Limit: 10 chars)
     - Email Address: Col 7 (Limit: 132 chars)
    """   
    # nedded to fix the  first search all then edit if not go to new entry
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLMEOME4&CustomizingObject=V_024&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87002131&Type=SSCUI"
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()
        
        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page)
            if await check_and_abort_if_locked(page):
                return
            for idx, target in enumerate(targets, start=1):
                # --------------------------------------------------------
                # LAYER 1: START-OF-ROW SAFETY RESET
                # --------------------------------------------------------
                webgui_frame = await get_webgui_frame(page)
                if webgui_frame:
                    await webgui_frame.evaluate("""() => {
                        const all = document.querySelectorAll('*');
                        for (let el of all) { if (el.scrollLeft > 0) el.scrollLeft = 0; }
                        document.documentElement.scrollLeft = 0;
                        document.body.scrollLeft = 0;
                    }""")
                    await page.keyboard.press("End")
                    await page.keyboard.press("Home")
                    await asyncio.sleep(1) 
                # --------------------------------------------------------
                pur_grp = target.get("Pur_Grp")
                desc = target.get("Description")
                tel_grp = target.get("Tel.No. Pur.Grp") or target.get("Tel_No_Pur_Grp")
                fax = target.get("Fax Number") or target.get("Fax")
                telephone = target.get("Telephone")
                extension = target.get("Extension")
                email = target.get("Email")
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING PUR GRP: {pur_grp}")
                requests = []
                if desc: 
                    requests.append({'col': 2, 'want': desc[:18], 'name': 'Description'})
                if tel_grp:
                    requests.append({'col': 3, 'want': tel_grp[:12], 'name': 'Tel. No. Pur Grp'})
                if fax:
                    # Ensuring style: 770 840 XXXX
                    digits = "".join(filter(str.isdigit, fax))
                    if len(digits) >= 10:
                        fax_formatted = f"{digits[:3]} {digits[3:6]} {digits[6:]}"[:31]
                    else:
                        fax_formatted = fax[:31]
                    requests.append({'col': 4, 'want': fax_formatted, 'name': 'Fax'})
                if telephone:
                    # Ensuring style: 770 840 XXXX
                    digits = "".join(filter(str.isdigit, telephone))
                    if len(digits) >= 10:
                        tel_formatted = f"{digits[:3]} {digits[3:6]} {digits[6:]}"[:30]
                    else:
                        tel_formatted = telephone[:30]
                    requests.append({'col': 5, 'want': tel_formatted, 'name': 'Telephone'})
                if extension:
                    requests.append({'col': 6, 'want': str(extension)[:10], 'name': 'Extension'})
                if email:
                    requests.append({'col': 7, 'want': email[:132], 'name': 'Email'})
                success = await find_and_set_material_groups(page, pur_grp, requests)
                if success:
                    LOGGER.info(f"SUCCESSFULLY UPDATED: {pur_grp}")
                else:
                    LOGGER.error(f"FAILED TO UPDATE: {pur_grp}")
                # --------------------------------------------------------
                # LAYER 2: IMMEDIATE RETURN AFTER EMAIL (As requested)
                # --------------------------------------------------------
                if webgui_frame:
                    await webgui_frame.evaluate("""() => {
                        const all = document.querySelectorAll('*');
                        for (let el of all) { if (el.scrollLeft > 0) el.scrollLeft = 0; }
                        document.documentElement.scrollLeft = 0;
                        document.body.scrollLeft = 0;
                    }""")
                    await page.keyboard.press("Home")
                    LOGGER.info("SCROLL RESET PERFORMED. WAITING FOR SAP TO SETTLE...")
                    await asyncio.sleep(1) # Stabilizer after Email
                # --------------------------------------------------------
            webgui = await get_webgui_frame(page)
            await execute_save_flow_fast(webgui, page)
            LOGGER.info("DONE. OPERATIONS COMPLETE.")
        
        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)
        
        finally:
            await safe_session_cleanup(browser, page)

JS_GET_TOLERANCE_DOM_IDS = """
() => {
    const mapping = {
        lower_limit: { absolute: null, percentage: null },
        upper_limit: { absolute: null, percentage: null },
        has_lower: false,
        has_upper: false,
        debug: {}
    };

    // ─── Helper: find visible leaf-ish elements by text ───
    const leafQuery = (pattern, exact) => {
        const out = [];
        const isRegex = pattern instanceof RegExp;
        document.querySelectorAll('span, label, div, td, legend, b, h3, h4, h5, p, a, strong').forEach(el => {
            const t = (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim();
            if (!t) return;
            const match = isRegex ? pattern.test(t) : (exact ? t === pattern : t.includes(pattern));
            if (match && el.offsetParent !== null && el.children.length <= 8) out.push(el);
        });
        return out;
    };

    // ─── Step 1: Section headers ───
    const lowerEls = leafQuery('Lower Limit', true);
    const upperEls = leafQuery('Upper Limit', true);
    let lowerY = -9999, upperY = 99999;
    if (lowerEls.length > 0) { lowerY = lowerEls[0].getBoundingClientRect().top; mapping.has_lower = true; }
    if (upperEls.length > 0) { upperY = upperEls[0].getBoundingClientRect().top; mapping.has_upper = true; }

    // ─── Step 2: Subsection headers ───
    const absoluteEls = leafQuery('Absolute', true);
    const percentageEls = leafQuery('Percentage', true);
    const hasAbsolute = absoluteEls.length > 0;
    const hasPercentage = percentageEls.length > 0;

    // ─── Step 3: Find all radio inputs and associate with labels ───
    // SAP WebGUI hides real <input type="radio"> (width:0,height:0) and overlays custom UI.
    // We must find them regardless of size, and also check for role="radio" and SAP classes.
    const allRadios = Array.from(document.querySelectorAll(
        'input[type="radio"], [role="radio"], .urRb input, .sapMRb input, span[ct="RB"]'
    ));
    const radioEntries = [];

    for (const radio of allRadios) {
        const rect = radio.getBoundingClientRect();
        // For hidden inputs, use the parent's rect instead
        let useRect = rect;
        if (rect.width === 0 || rect.height === 0) {
            let p = radio.parentElement;
            for (let i = 0; i < 5 && p; i++) {
                const pr = p.getBoundingClientRect();
                if (pr.width > 0 && pr.height > 0) { useRect = pr; break; }
                p = p.parentElement;
            }
        }

        let labelText = '';

        // Method 1: <label for="id">
        if (radio.id) {
            const lbl = document.querySelector('label[for="' + radio.id + '"]');
            if (lbl) labelText = (lbl.innerText || lbl.textContent || '').trim();
        }

        // Method 2 (PRIORITY): proximity search — find nearest label by position.
        // This must run BEFORE sibling traversal because sibling scanning picks the
        // first label in DOM order, which is always "Do Not Check" for BOTH radios!
        if (!labelText) {
            const allLabels = document.querySelectorAll('label, span, td, b, strong, div');
            let bestDist = 250;
            for (const lbl of allLabels) {
                const lr = lbl.getBoundingClientRect();
                const t = (lbl.innerText || lbl.textContent || '').replace(/\\s+/g, ' ').trim();
                const isMatch = (t === 'Do Not Check' || t === 'Check Limit' || t.includes('Check'));
                if (!isMatch) continue;
                if (lbl.children.length > 5) continue; // skip containers
                
                const normalizedText = t.toLowerCase().includes('do not') ? 'Do Not Check' : 'Check Limit';
                
                const yDist = Math.abs(lr.top - useRect.top);
                // Radios and labels should be on the same vertical alignment (within 35px)
                if (yDist < 35) {
                    const dist = Math.sqrt(
                        Math.pow(lr.left - useRect.left, 2) + Math.pow(lr.top - useRect.top, 2)
                    );
                    if (dist < bestDist) {
                        bestDist = dist;
                        labelText = normalizedText;
                    }
                }
            }
        }

        // Method 3 (FALLBACK): traverse up ancestors, check siblings for label text
        if (!labelText) {
            let p = radio.parentElement;
            for (let i = 0; i < 8 && p; i++) {
                const siblings = Array.from(p.children);
                for (const sib of siblings) {
                    if (sib === radio || sib.contains(radio)) continue;
                    const t = (sib.innerText || sib.textContent || '').trim();
                    if (t === 'Do Not Check' || t === 'Check Limit') { labelText = t; break; }
                }
                if (labelText) break;
                if (p.nextElementSibling) {
                    const t = (p.nextElementSibling.innerText || p.nextElementSibling.textContent || '').trim();
                    if (t === 'Do Not Check' || t === 'Check Limit') { labelText = t; break; }
                }
                p = p.parentElement;
            }
        }

        if (!labelText) continue;
        radioEntries.push({
            id: radio.id, label: labelText,
            x: useRect.left + useRect.width/2, y: useRect.top + useRect.height/2,
            checked: radio.checked
        });
    }

    // ─── Step 4: Classify each radio entry by section + subsection ───
    const classifyY = (y) => {
        if (mapping.has_lower && mapping.has_upper) return y < upperY ? 'lower_limit' : 'upper_limit';
        if (mapping.has_lower) return 'lower_limit';
        if (mapping.has_upper) return 'upper_limit';
        return null;
    };
    const classifyX = (x) => {
        if (hasAbsolute && !hasPercentage) return 'absolute';
        if (hasPercentage && !hasAbsolute) return 'percentage';
        if (!hasAbsolute && !hasPercentage) return 'percentage';
        return x < window.innerWidth * 0.45 ? 'absolute' : 'percentage';
    };

    for (const entry of radioEntries) {
        const section = classifyY(entry.y);
        const subsection = classifyX(entry.x);
        if (!section || !subsection) continue;
        if (!mapping[section][subsection]) mapping[section][subsection] = {};
        if (entry.label === 'Do Not Check') {
            mapping[section][subsection].do_not_check_id = entry.id;
            mapping[section][subsection].do_not_check_checked = entry.checked;
        } else if (entry.label === 'Check Limit') {
            mapping[section][subsection].check_limit_id = entry.id;
            mapping[section][subsection].check_limit_checked = entry.checked;
        }
    }

    // ─── Step 5: Find input fields ───
    // Broad search for inputs
    const allInputs = Array.from(document.querySelectorAll('input:not([type=\"hidden\"]), [role=\"textbox\"]'));
    const textInputs = allInputs.filter(i => {
        const r = i.getBoundingClientRect();
        return r.width > 20 && r.height > 5;
    });
    
    // *** KEY FIX: Filter out header inputs (Tolerance Key, Company Code, Amounts in) ***
    // These are ABOVE the limit sections. Only keep inputs that are below the 
    // first section header (lowerY or upperY).
    const sectionStartY = mapping.has_lower ? lowerY : (mapping.has_upper ? upperY : 99999);
    const sectionInputs = textInputs.filter(inp => {
        const ir = inp.getBoundingClientRect();
        return ir.top >= sectionStartY - 10;  // must be at or below the section header
    });
    
    mapping.debug_inputs = [];
    for (const inp of textInputs) {
        const ir = inp.getBoundingClientRect();
        mapping.debug_inputs.push({
            id: inp.id, y: Math.round(ir.top), x: Math.round(ir.left),
            w: Math.round(ir.width), val: inp.value,
            inSection: ir.top >= sectionStartY - 10
        });
    }

    // Label variations — search ONLY for "Value:" label text (exact word)
    const valueLabelEls = leafQuery('Value:', false);
    // Fallback: also check for "Value" without colon but within section area
    if (valueLabelEls.length === 0) {
        const fallback = leafQuery('Value', true);
        for (const fl of fallback) {
            if (fl.getBoundingClientRect().top >= sectionStartY - 10) {
                valueLabelEls.push(fl);
            }
        }
    }
    
    const pctLabelEls = [
        ...leafQuery(/Tolerance Limit %/i, false), ...leafQuery(/Limit %/i, false), ...leafQuery(/%/i, true)
    ].filter(el => el.getBoundingClientRect().top >= sectionStartY - 10);

    const findInputForLabel = (labelEl) => {
        const lr = labelEl.getBoundingClientRect();
        let best = null, bestDist = 600;
        // ONLY search within section inputs, NOT header inputs
        for (const inp of sectionInputs) {
            const ir = inp.getBoundingClientRect();
            // Search horizontally (input to the right of label)
            if (Math.abs(ir.top - lr.top) < 30 && ir.left >= lr.left - 20) {
                const d = ir.left - lr.left;
                if (d > 0 && d < bestDist) { bestDist = d; best = inp; }
            }
            // Search vertically (input below label)
            if (!best && ir.top > lr.top && ir.top - lr.top < 50 && Math.abs(ir.left - lr.left) < 150) {
                const d = ir.top - lr.top;
                if (d < bestDist) { bestDist = d; best = inp; }
            }
        }
        return best;
    };

    // Try label-based matching
    for (const lbl of valueLabelEls) {
        const inp = findInputForLabel(lbl);
        if (!inp) continue;
        const section = classifyY(lbl.getBoundingClientRect().top);
        if (section && mapping[section].absolute) {
            mapping[section].absolute.input_id = inp.id || null;
            mapping[section].absolute.current_value = inp.value || '';
        }
    }
    for (const lbl of pctLabelEls) {
        const inp = findInputForLabel(lbl);
        if (!inp) continue;
        const section = classifyY(lbl.getBoundingClientRect().top);
        if (section && mapping[section].percentage) {
            mapping[section].percentage.input_id = inp.id || null;
            mapping[section].percentage.current_value = inp.value || '';
        }
    }

    // ─── Fallback: Proximity to Radios ───
    for (const sec of ['lower_limit', 'upper_limit']) {
        for (const sub of ['absolute', 'percentage']) {
            const secData = mapping[sec][sub];
            if (secData && !secData.input_id) {
                const refId = secData.check_limit_id || secData.do_not_check_id;
                if (!refId) continue;
                const refEl = document.getElementById(refId);
                if (!refEl) continue;
                let refRect = refEl.getBoundingClientRect();
                if (refRect.width === 0) {
                   let p = refEl.parentElement;
                   for(let i=0;i<5&&p;i++){
                       const pr = p.getBoundingClientRect();
                       if(pr.width > 0){ refRect=pr; break; }
                       p=p.parentElement;
                   }
                }
                
                let best = null, bestD = 400;
                // ONLY search section inputs (below header)
                for (const inp of sectionInputs) {
                    const ir = inp.getBoundingClientRect();
                    const dist = Math.sqrt(Math.pow(ir.left - refRect.left, 2) + Math.pow(ir.top - refRect.top, 2));
                    // Input must be BELOW the radio (Value box is always below Check Limit)
                    if (dist < bestD && ir.top >= refRect.top - 10 && ir.top < refRect.top + 120) {
                        bestD = dist;
                        best = inp;
                    }
                }
                if (best) {
                    secData.input_id = best.id || null;
                    secData.current_value = best.value || '';
                }
            }
        }
    }

    mapping.debug = {
        total_radios: allRadios.length, classified_radios: radioEntries.length,
        text_inputs: textInputs.length, section_inputs: sectionInputs.length,
        lower_y: lowerY, upper_y: upperY, section_start_y: sectionStartY,
        viewport_width: window.innerWidth
    };
    return mapping;
}
"""

async def filter_max_length(text: str, max_len: int = 7) -> str:
    if not text:
        return ""
    text_str = str(text)
    if len(text_str) > max_len:
        LOGGER.warning(f"Value '{text_str}' exceeds length limit {max_len}. Truncating.")
        text_str = text_str[:max_len]
    return text_str

async def _click_radio_by_id(frame, page, radio_id):
    """Click a radio button using Playwright's REAL mouse click on the visible parent.
    SAP WebGUI hides real <input type=radio> (0x0 size) behind custom UI.
    JS .click() does NOT trigger SAP's UI5 event handlers — we MUST use
    Playwright's native click which generates real mousedown/mouseup events."""
    if not radio_id:
        LOGGER.warning("No radio ID provided, cannot click.")
        return False
    try:
        # Step 1: Use JS to find the visible parent container and mark it
        found = await frame.evaluate(f"""() => {{
            // Clear any old markers
            document.querySelectorAll('[data-pw-radio-target]').forEach(
                e => e.removeAttribute('data-pw-radio-target')
            );
            const radio = document.getElementById('{radio_id}');
            if (!radio) return false;
            
            // Find the closest VISIBLE ancestor
            let clickTarget = radio;
            let p = radio.parentElement;
            for (let i = 0; i < 8 && p; i++) {{
                const r = p.getBoundingClientRect();
                if (r.width > 5 && r.height > 5 && r.width < 300) {{
                    clickTarget = p;
                    break;
                }}
                p = p.parentElement;
            }}
            
            // Mark it so Playwright can find it
            clickTarget.setAttribute('data-pw-radio-target', 'true');
            return true;
        }}""")
        
        if not found:
            LOGGER.warning(f"Radio element #{radio_id} not found in DOM.")
            return False
        
        # Step 2: Use Playwright's REAL click on the marked element
        target = frame.locator('[data-pw-radio-target="true"]').first
        if await target.count() > 0:
            await target.click(force=True)
            LOGGER.info(f"Radio #{radio_id} clicked via Playwright native click.")
            await asyncio.sleep(1)
            return True
        
        # Step 3: Fallback — try clicking the radio itself with force
        radio_loc = frame.locator(f"[id='{radio_id}']").first
        if await radio_loc.count() > 0:
            await radio_loc.click(force=True)
            LOGGER.info(f"Radio #{radio_id} clicked directly (force).")
            await asyncio.sleep(1)
            return True
        
        LOGGER.warning(f"Could not locate radio #{radio_id} for Playwright click.")
        return False
    except Exception as e:
        LOGGER.warning(f"Failed to click radio {radio_id}: {e}")
        return False

async def _type_into_input(frame, page, input_id, value):
    """Type a value into a SAP UI5 input field using Playwright's native interaction.
    
    SAP UI5 data binding requires real user-like events (mousedown, focus, keydown, input, keyup).
    Directly setting el.value via JS bypasses the UI5 MVC model and the value reverts on save.
    This function clicks the field, selects all, and physically types via Playwright keyboard.
    """
    if not input_id:
        LOGGER.warning("No input ID provided, cannot type.")
        return False
    try:
        inp_loc = frame.locator(f"[id='{input_id}']")
        if await inp_loc.count() == 0:
            LOGGER.warning(f"Input field #{input_id} not found in DOM.")
            return False

        LOGGER.info(f"    Writing value '{value}' into input: #{input_id}")

        # 1. Scroll element into center of viewport
        await frame.evaluate(f"""() => {{
            const el = document.getElementById('{input_id}');
            if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center', inline: 'center' }});
        }}""")
        await asyncio.sleep(0.3)

        # 2. Click the field to give it focus
        try:
            await inp_loc.first.click(force=True, timeout=3000)
        except Exception:
            # JS fallback click if Playwright can't reach it
            await frame.evaluate(f"() => document.getElementById('{input_id}')?.click()")
        await asyncio.sleep(0.4)

        # 3. Select all existing content and delete it
        await page.keyboard.press("Control+a")
        await asyncio.sleep(0.15)
        await page.keyboard.press("Delete")
        await asyncio.sleep(0.15)

        # 4. Type the new value character-by-character so SAP UI5 fires all change events
        await page.keyboard.type(str(value), delay=0)
        await asyncio.sleep(0.3)

        # 5. Press Tab to commit the value and move focus away (triggers UI5 blur/change)
        await page.keyboard.press("Tab")
        await asyncio.sleep(0.2)

        LOGGER.info(f"    ✔ Value '{value}' typed successfully.")
        return True
    except Exception as e:
        LOGGER.warning(f"Failed to type into #{input_id}: {e}")
        return False

@time_it
async def Set_Tolerance_limits_101947(targets: list[dict]):
    """Main flow for SSCUI 101947 (Set Tolerance Limits).

    Handles all combinations of:
    - Lower Limit / Upper Limit (or both)
    - Absolute / Percentage (or both) within each limit
    - Check Limit / Do Not Check radio selection
    - Value entry for absolute (Value:) and percentage (Tolerance Limit %:)

    Target format:
        {
            "company_code": "9000",
            "tolerance_key": "VP",
            "lower_limit": {
                "absolute":   {"check": True/False, "value": "10.25"},
                "percentage": {"check": True/False, "value": "20.00"}
            },
            "upper_limit": {
                "absolute":   {"check": True/False, "value": "5.11"},
                "percentage": {"check": True/False, "value": "5.00"}
            }
        }
    Omit any section/subsection that doesn't exist on the screen.
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUOLMROMR6&CustomizingObject=V_169G&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87002392&Type=SSCUI"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            LOGGER.info(f"OPENING: {url}")
            await install_lock_watcher(page)
            await page.goto(url, wait_until="commit", timeout=60000)
            try:
                await login(page)
            except Exception as e:
                LOGGER.error(str(e))
                return
            
            webgui = await get_webgui_frame(page)
            try:
                await webgui.locator(".sapUiTable, .lsTable").first.wait_for(state="visible", timeout=5000)
            except: pass
            await asyncio.sleep(0.2)

            # Determine working frame (WebGUI iframe or page itself)
            frame = await get_webgui_frame(page) or page
            LOGGER.info(f"Working frame type: {'WebGUI iframe' if frame != page else 'main page'}")

            for idx, target in enumerate(targets, start=1):
                # Guard: stop if browser was closed
                if page.is_closed():
                    LOGGER.error("Browser/page was closed. Stopping.")
                    break

                company_code = str(target.get("company_code", ""))
                tolerance_key = str(target.get("tolerance_key", ""))
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING CoCd={company_code}, TlKy={tolerance_key}")

                # Re-acquire frame reference (it may change after F3 navigation)
                frame = await get_webgui_frame(page) or page

                # ════════════════════════════════════════════════════════════
                # STEP 1: Navigate to the target entry in the list view
                # ════════════════════════════════════════════════════════════
                entered_details = False

                # Strategy A: Use 'Position...' button to scroll to row
                try:
                    # SAP may render '...' as literal dots or Unicode ellipsis '…'
                    pos_btn = frame.locator("text=/Position/i").first
                    if await pos_btn.is_visible(timeout=3000):
                        LOGGER.info("Found 'Position...' button. Clicking...")
                        await pos_btn.click(force=True)
                        await asyncio.sleep(1)

                        # Position dialog: enter Company Code and Tolerance Key
                        dialog = frame.locator("[role='dialog'], .urPW, .urMsgBox").first
                        if await dialog.is_visible(timeout=3000):
                            inputs = dialog.locator("input[type='text']")
                            input_count = await inputs.count()
                            LOGGER.info(f"Position dialog has {input_count} input fields")

                            if input_count >= 2:
                                # Field order: Company Code first, then Tolerance Key
                                await inputs.nth(0).click()
                                await page.keyboard.press("Control+A")
                                await page.keyboard.press("Backspace")
                                await page.keyboard.type(company_code, delay=0)
                                await asyncio.sleep(0.3)

                                await inputs.nth(1).click()
                                await page.keyboard.press("Control+A")
                                await page.keyboard.press("Backspace")
                                await page.keyboard.type(tolerance_key, delay=0)
                                await asyncio.sleep(0.3)
                            elif input_count == 1:
                                await inputs.nth(0).click()
                                await page.keyboard.press("Control+A")
                                await page.keyboard.press("Backspace")
                                await page.keyboard.type(company_code, delay=0)
                                await asyncio.sleep(0.3)

                            # Confirm dialog
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(1)

                            # Check if dialog is still open, click Continue if so
                            try:
                                cont_btn = dialog.locator("[title*='Continue'], [title*='Copy'], button:has-text('Continue')").first
                                if await cont_btn.is_visible(timeout=1000):
                                    await cont_btn.click(force=True)
                                    await asyncio.sleep(1)
                            except:
                                pass
                            LOGGER.info("Position dialog completed. Row should now be visible.")
                        else:
                            LOGGER.warning("Position dialog did not appear after clicking button.")
                    else:
                        LOGGER.warning("Position... button not found on page.")
                except Exception as e:
                    LOGGER.warning(f"Position dialog approach failed: {e}")

                # Strategy B: Find row in table and navigate to details
                if not entered_details:
                    try:
                        await asyncio.sleep(1)
                        # Use JS exact cell matching — NOT substring matching!
                        # has_text="AN" would match "Amount..." descriptions, causing wrong row selection.
                        row = None
                        for scroll_attempt in range(10):
                            # JS: find row where individual cells contain exact company_code and tolerance_key
                            row_idx = await frame.evaluate(f"""() => {{
                                const rows = document.querySelectorAll('tr');
                                for (let i = 0; i < rows.length; i++) {{
                                    const cells = rows[i].querySelectorAll('td');
                                    if (cells.length < 3) continue;
                                    let hasCC = false, hasTK = false;
                                    for (const cell of cells) {{
                                        const txt = (cell.innerText || cell.textContent || '').trim();
                                        if (txt === '{company_code}') hasCC = true;
                                        if (txt === '{tolerance_key}') hasTK = true;
                                    }}
                                    if (hasCC && hasTK) return i;
                                }}
                                return -1;
                            }}""")
                            
                            if row_idx >= 0:
                                # Click the first cell of the found row to select it
                                await frame.evaluate(f"""() => {{
                                    const rows = document.querySelectorAll('tr');
                                    const row = rows[{row_idx}];
                                    if (row) {{
                                        const firstCell = row.querySelector('td');
                                        if (firstCell) firstCell.click();
                                    }}
                                }}""")
                                LOGGER.info(f"Found and clicked exact row for {company_code}/{tolerance_key} (scroll attempt {scroll_attempt})")
                                await asyncio.sleep(1)
                                row = True  # Flag that we found the row
                                break
                            
                            # Row not found in current view — scroll down
                            # Page down in the table to load more rows
                            LOGGER.info(f"Row not visible, pressing Page Down (attempt {scroll_attempt + 1}/10)...")
                            await page.keyboard.press("PageDown")
                            await asyncio.sleep(1)
                        
                        if row:
                            # Row already clicked by JS above. Now click Details button.
                            details_clicked = False
                            for details_text in ["Details", "Detail"]:
                                try:
                                    det_btn = frame.locator(f"text='{details_text}'").first
                                    if await det_btn.is_visible(timeout=1000):
                                        await det_btn.click(force=True)
                                        details_clicked = True
                                        LOGGER.info(f"Clicked '{details_text}' button.")
                                        break
                                except:
                                    continue

                            if not details_clicked:
                                # Fallback: try F2 or Ctrl+Shift+F2 to open details
                                LOGGER.info("Details button not found. Trying Ctrl+Shift+F2...")
                                await page.keyboard.press("Control+Shift+F2")

                            await asyncio.sleep(3)

                            # Verify we're on details screen
                            frame = await get_webgui_frame(page) or page
                            try:
                                still_list = await frame.locator("text=/Position/i").first.is_visible(timeout=2000)
                            except:
                                still_list = False

                            if not still_list:
                                entered_details = True
                                LOGGER.info("Successfully entered details view.")
                            else:
                                LOGGER.error(f"Cannot enter details for {company_code}/{tolerance_key}. Skipping.")
                                continue
                        else:
                            LOGGER.warning(f"Row {company_code}/{tolerance_key} NOT FOUND. Attempting NEW ENTRY creation...")
                            
                            # Fallback logic to click New Entries
                            clicked = False
                            for sel in ["a:has-text('New Entries')", "button:has-text('New Entries')", "span:has-text('New Entries')"]:
                                try:
                                    btn = frame.locator(sel).first
                                    if await btn.is_visible(timeout=500):
                                        await btn.click(force=True)
                                        clicked = True
                                        await asyncio.sleep(1)
                                        break
                                except: pass
                            
                            if not clicked:
                                LOGGER.error("'New Entries' button not found.")
                                continue
                            
                            frame = await get_webgui_frame(page) or page
                            await asyncio.sleep(1)
                            
                            LOGGER.info("Filling TlKy and CoCd for New Entry...")
                            # When New Entries opens on this screen, SAP automatically focuses the Tolerance Key first.
                            # We type the Tolerance Key, hit Tab, then type Company Code.
                            
                            await page.keyboard.press("Control+a")
                            await page.keyboard.press("Backspace")
                            await page.keyboard.type(tolerance_key, delay=0)
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.2)
                            
                            await page.keyboard.press("Control+a")
                            await page.keyboard.press("Backspace")
                            await page.keyboard.type(company_code, delay=0)
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(1)
                            
                            # Validate we are on the form now
                            frame = await get_webgui_frame(page) or page
                            try:
                                still_list = await frame.locator("text=/Position/i").first.is_visible(timeout=2000)
                            except:
                                still_list = False
                                
                            if still_list:
                                LOGGER.error("Failed to enter details view after New Entry.")
                                continue
                                
                            LOGGER.info("Successfully entered New Entry Details. Falling through to limit mapping...")
                    except Exception as e:
                        LOGGER.error(f"Failed to navigate to details: {e}", exc_info=True)
                        continue

                # ════════════════════════════════════════════════════════════
                # STEP 2: Read DOM structure on the details screen
                # ════════════════════════════════════════════════════════════
                await asyncio.sleep(1)
                elem_map = await frame.evaluate(JS_GET_TOLERANCE_DOM_IDS)
                LOGGER.info(f"DOM MAP DEBUG: {json.dumps(elem_map.get('debug', {}))}")
                LOGGER.info(f"  has_lower={elem_map.get('has_lower')}, has_upper={elem_map.get('has_upper')}")
                # Log ALL input candidates so we can verify the correct one is selected
                for di in elem_map.get('debug_inputs', []):
                    LOGGER.info(f"  INPUT CANDIDATE: id={di.get('id')}, y={di.get('y')}, val={di.get('val')}, inSection={di.get('inSection')}")
                # Log the radio ID mappings for each section so we can verify correctness
                for lt in ["lower_limit", "upper_limit"]:
                    sec = elem_map.get(lt, {})
                    if sec:
                        for st in ["absolute", "percentage"]:
                            sub = sec.get(st)
                            if sub:
                                LOGGER.info(f"  {lt}.{st}: check_limit_id={sub.get('check_limit_id')}, do_not_check_id={sub.get('do_not_check_id')}, input_id={sub.get('input_id')}, current_value={sub.get('current_value')}")

                # ════════════════════════════════════════════════════════════
                # STEP 3: Process each limit type and subsection
                # ════════════════════════════════════════════════════════════
                for limit_type in ["lower_limit", "upper_limit"]:
                    limit_config = target.get(limit_type)
                    if not limit_config:
                        continue

                    for sub_type in ["absolute", "percentage"]:
                        sub_config = limit_config.get(sub_type)
                        if not sub_config:
                            continue

                        dom_info = elem_map.get(limit_type, {}).get(sub_type)
                        if not dom_info:
                            LOGGER.warning(f"No DOM elements found for {limit_type}.{sub_type} — section may not exist on this screen. Skipping.")
                            continue

                        should_check = sub_config.get("check", False)
                        value = str(sub_config.get("value", ""))
                        
                        LOGGER.info(f"  {limit_type}.{sub_type}: check={should_check}, value='{value}'")
                        
                        # Flow: Always attempt to write value if provided.
                        # If check is False, we temporarily select Check Limit to enable input, type, then select Do Not Check.
                        
                        check_id = dom_info.get("check_limit_id")
                        dnc_id = dom_info.get("do_not_check_id")
                        input_id = dom_info.get("input_id")
                        
                        if value:
                            if check_id:
                                LOGGER.info(f"    Temporarily selecting 'Check Limit' to ensure input is enabled: #{check_id}")
                                await _click_radio_by_id(frame, page, check_id)
                                await asyncio.sleep(0.2)
                            
                            if input_id:
                                # Apply length filtering
                                val = await filter_max_length(value)
                                LOGGER.info(f"    Writing value '{val}' to input: #{input_id}")
                                await _type_into_input(frame, page, input_id, val)
                            else:
                                LOGGER.warning(f"    No input field found for {limit_type}.{sub_type}")
                        
                        # Finalize radio selection based on user request (should_check)
                        if should_check:
                            if check_id:
                                # Only click if we didn't just click it above
                                if not value:
                                    LOGGER.info(f"    Setting final state to 'Check Limit': #{check_id}")
                                    await _click_radio_by_id(frame, page, check_id)
                        else:
                            if dnc_id:
                                LOGGER.info(f"    Setting final state to 'Do Not Check': #{dnc_id}")
                                await _click_radio_by_id(frame, page, dnc_id)
                            else:
                                LOGGER.warning(f"    No 'Do Not Check' radio found for {limit_type}.{sub_type}")

                # ════════════════════════════════════════════════════════════
                # STEP 4: Save changes
                # ════════════════════════════════════════════════════════════
                LOGGER.info("Saving changes for current target...")

                # Try clicking the visible Save button first (Fiori footer)
                save_clicked = False
                try:
                    save_btn = frame.locator("button:has-text('Save'), [title='Save'], .sapMBtn:has-text('Save')").first
                    if await save_btn.is_visible(timeout=2000):
                        await save_btn.click(force=True)
                        save_clicked = True
                        LOGGER.info("Clicked visible Save button.")
                        await asyncio.sleep(3)
                except:
                    pass

                if not save_clicked:
                    # Fallback: Ctrl+S
                    await page.keyboard.press("Control+s")
                    LOGGER.info("Pressed Ctrl+S.")
                    await asyncio.sleep(3)

                # Handle transport request / confirmation dialogs
                await handle_sap_confirmation_dialogs(page, timeout_ms=3000)
                await asyncio.sleep(1)

                # Check for error messages
                try:
                    status_msg = frame.locator("[id*='msgarea'], .lsMessageBar, [role='status']")
                    if await status_msg.count() > 0 and await status_msg.first.is_visible(timeout=1000):
                        msg_text = await status_msg.first.inner_text()
                        if msg_text.strip():
                            LOGGER.info(f"SAP Status Message: {msg_text.strip()}")
                            msg_lower = msg_text.lower()
                            if "error" in msg_lower or "exceeded" in msg_lower or "does not exist" in msg_lower or "not allowed" in msg_lower:
                                LOGGER.warning(f"SAP ERROR ON SAVE: {msg_text}")
                except:
                    pass

                # ════════════════════════════════════════════════════════════
                # STEP 5: Return to list view for next target
                # ════════════════════════════════════════════════════════════
                LOGGER.info("Returning to list view (F3)...")
                await page.keyboard.press("F3")
                await asyncio.sleep(4)

                # Verify we're back on the list view — do NOT press F3 again
                # as that would exit the entire transaction!
                frame = await get_webgui_frame(page) or page
                try:
                    pos_visible = await frame.locator("text=/Position/i").first.is_visible(timeout=3000)
                    if pos_visible:
                        LOGGER.info("Back on list view.")
                    else:
                        LOGGER.warning("Position button not visible — we might be stuck due to an error.")
                        LOGGER.info("Pressing F12 (Cancel) to force exit details view...")
                        await page.keyboard.press("F12")
                        await asyncio.sleep(1)
                        
                        try:
                            # Handle "Data will be lost" confirmation
                            for f in page.frames:
                                yes_btn = f.locator("button:has-text('Yes'), [title*='Yes'], .sapMBtn:has-text('Yes')").first
                                if await yes_btn.is_visible(timeout=500):
                                    await yes_btn.click(force=True)
                                    await asyncio.sleep(1)
                                    break
                        except:
                            pass
                        
                        # Sometimes simple F3 works after clearing error, try once more just in case
                        await page.keyboard.press("F3")
                        await asyncio.sleep(3)
                except:
                    LOGGER.warning("Could not verify list view state.")

            LOGGER.info("ALL TARGETS PROCESSED. OPERATIONS COMPLETE.")

        except Exception as e:
            LOGGER.error(f"Flow error: {e}", exc_info=True)

        finally:
            await safe_session_cleanup(browser, page)

if __name__ == "__main__":
    pass
    # # LOGGER.info("▶️ STARTING TEST: SSCUI 102911")
    # targets_102911 = [
    #     {"target_name": "NB",  "opt1_val": True, "opt2_val": False},
    #     {"target_name": "NBS", "opt1_val": False, "opt2_val": False},
    #     {"target_name": "RV",  "opt1_val": False, "opt2_val": True},
    # ]
    # asyncio.run(activate_purchase_requisition_workflow_102911(targets=targets_102911))

    # # # 2. SSCUI 102888 — Configure Purchase Requisition
    # # LOGGER.info("▶️ STARTING TEST: SSCUI 102888")
    # targets_102888 = [
     
    #     {"target_name": "Currency Key",              "ss_proc": False, "prf_proc": False},
    #     {"target_name": "Price Unit",                "ss_proc": False,  "prf_proc": False},
    #     {"target_name": "Quantity of Items",         "ss_proc": False,  "prf_proc": False},
           
    #     {"target_name": "Purchase Requisition Price", "ss_proc": False, "prf_proc": False},
    # ]
    # asyncio.run(configure_purchase_requisition_102888(targets=targets_102888))

    # # # 3. SSCUI 101097 — Activate PO Flexible Workflow  CHECK THIS 
    # # LOGGER.info("▶️ STARTING TEST: SSCUI 101097")
    # targets_101097 = [
    #     {"Type": "NB",   "Scenario_based_workflow": True},
    #     {"Type": "NB2",  "Scenario_based_workflow": False},
    #     {"Type": "NBAI", "Scenario_based_workflow": False},
    #     {"Type": "NBIC", "Scenario_based_workflow": False},
    # ]
    # asyncio.run(activate_po_flexible_workflow_101097(targets=targets_101097))
    
    # # # 4. SSCUI 103345 — PO Workflow Restart   MAKE IT MORE SPEED
    # # LOGGER.info("▶️ STARTING TEST: SSCUI 103345")
    # targets_103345 = [
    #     {"Purchase_Order_Attributes": "Company Code",             "Enable": True,  "Restart_Type": "Always Restart"},
    #     {"Purchase_Order_Attributes": "Purchasing Group",          "Enable": False,  "Restart_Type": ""},
    #     {"Purchase_Order_Attributes": "Purchasing Organization",   "Enable": True, "Restart_Type": "Conditional Restart"},
    #     {"Purchase_Order_Attributes": "Total Net Order Value",     "Enable": True, "Restart_Type": "Conditional Restart"},
    #     {"Purchase_Order_Attributes": "Incoterms",                "Enable": False,  "Restart_Type": ""},
    #     {"Purchase_Order_Attributes": "Outline Agreement",        "Enable": True,  "Restart_Type": "Always Restart"},
    #     {"Purchase_Order_Attributes": "Material Group",           "Enable": True,  "Restart_Type": "Always Restart"},
    #     {"Purchase_Order_Attributes": "Currency",                 "Enable": False,  "Restart_Type": ""},
    #     {"Purchase_Order_Attributes": "Plant",                    "Enable": True,  "Restart_Type": "Always Restart"},
    # ]
    # asyncio.run(configure_po_workflow_restart_conditions_103345(targets=targets_103345))

    # # 5. SSCUI 101098 — Activate Flexible Workflow for Supplier Invoices
    # LOGGER.info("▶️ STARTING TEST: SSCUI 101098")
    # asyncio.run(supply_invoices_101098(payment_block=True, release_completed=False, check_auth=False))

    # # 6. SSCUI 101247 — Document Types Contract Change
    # LOGGER.info("▶️ STARTING TEST: SSCUI 101247")
    # targets_101247 = [
    #     {"type_code": "MK", "Scenario_based_workflow": True},
    #     {"type_code": "CWK", "Scenario_based_workflow": True}
    # ]
    # asyncio.run(Document_Types_Contract_Change_101247(targets=targets_101247))

    # # 7. SSCUI 102665 — Define Material Groups
    # LOGGER.info("▶️ STARTING TEST: SSCUI 102665 (Create)")
    # target_new_102665= [
    #     {
    #         "Matl_Group":          "W1201",
    #         "Material_Group_Desc": "Water boiling",
    #         "AGrp":                "1538",
    #         "DUW":                 "BTU",
    #         "Description_2":       "Water for drinking",
    #     },
    #     {
    #         "Matl_Group":          "W1202",
    #         "Material_Group_Desc": "Water at 100 celcius",
    #         "AGrp":                "9538",
    #         "DUW":                 "D",
    #         "Description_2":       "Water at 100 celcius",
    #     }
    # ]
    # asyncio.run(Create_New_Material_Group_102665(targets=target_new_102665))

    # LOGGER.info("▶️ STARTING TEST: SSCUI 102665 (Edit)")
    # target_existing_102665 = [
    #     {
    #         "Matl_Group":          "P000",
    #         "Material_Group_Desc": "Contracter",
    #         "AGrp":                "5437",
    #         "DUW":                 "D",
    #         "Description_2":       "Contracter",
    #     },
    # ]
    # asyncio.run(Edit_Existing_Material_Group_102665(targets=target_existing_102665))

    # # 8. SSCUI 101602 — Entry Aids for Items Without a Material Master    tets again
    # LOGGER.info("▶️ STARTING TEST: SSCUI 101602")
    # targets_101602 = [
    #     {
    #         "Mat_Grp": "GRP001",
    #         "Mat_Grp_Descr": "Raw Materials items",
    #         "ValCl": "3000",
    #         "PurValK": "1",
    #     },
    # ]
    # asyncio.run(Entry_Aids_for_Items_Without_a_Material_Master_101602(targets=targets_101602))

    # # 9. SSCUI 105939 — Maintain Purchasing Organization  CHECK THIS 
    # LOGGER.info("▶️ STARTING TEST: SSCUI 105939")
    # targets_105939 = [
    #     {"POrg": "1001", "Description": "Automated Testing Organization dected"},
    # ]
    # asyncio.run(maintain_purchasing_organization_105939(targets=targets_105939))

    ## 10. SSCUI 102126 — Assign Shipping Points
    # LOGGER.info("▶️ STARTING TEST: SSCUI 102126")
    # targets_102126 = [
    #     {
    #         "Shipping_Condition":       "01",
    #         "Loading_Group":            "0001",
    #         "Plant":                    "1002",
    #         "Proposed_Shipping_Point":  "1002",
    #         "Alternative_Shipping_Points": ["1002","", "", "", "", "1002", "1002", "1002", "1002", "", "1002"],
    #     },
    # ]
    # asyncio.run(Assign_Shipping_Points_102126(targets=targets_102126))

    # 11. SSCUI 102130 — Assign Picking Locations
    LOGGER.info("▶️ STARTING TEST: SSCUI 102130 (Create)")
    target_new_102130 = [
        {
            "ShPt": "1003",
            "Plnt": "1003",
            "SC":   "10",
            "Stor": "FG01",
        },
        {
            "ShPt": "9001",
            "Plnt": "9001",
            "SC":   "20",
            "Stor": "RM01",
        },
    ]
    asyncio.run(Create_New_Picking_Locations_102130(targets=target_new_102130))

    # LOGGER.info("▶️ STARTING TEST: SSCUI 102130 (Edit)")  
    #    # CHECK  THIS

    # #12. SSCUI 102914 — Create Purchasing Groups  # CHECK THIS
    # LOGGER.info("▶️ STARTING TEST: SSCUI 102914")
    # targets_102914 = [
    #     {
    #         "Pur_Grp": "600",
    #         "Description": "Assets",
    #         "Email": "p600@example.com",
    #         "Fax Number": "12345789",
    #         "Telephone": "6586987456",
    #         "Extension": "93",
    #         "Tel.No. Pur.Grp": "99"
    #     },
    # ]
    # asyncio.run(create_purchase_groups_102914(targets=targets_102914))

    # 13. SSCUI 101947 — Set Tolerance Limits
    # LOGGER.info("▶️ STARTING TEST: SSCUI 101947")
    # targets_101947 = [
    #     {
    #         "company_code":  "1010",
    #         "tolerance_key": "VP",
    #         "lower_limit": {
    #             "absolute":   {"check": True,  "value": "10.00"},
    #             "percentage": {"check": True,  "value": "5.00"},
    #         },
    #         "upper_limit": {
    #             "absolute":   {"check": False, "value": ""},
    #             "percentage": {"check": True,  "value": "10.00"},
    #         },
    #     },
    # ]
    # asyncio.run(Set_Tolerance_limits_101947(targets=targets_101947))

