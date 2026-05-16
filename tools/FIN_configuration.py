

import asyncio
import json
import logging
import os
import re
import sys\
from playwright.async_api import async_playwright
# Add parent directory to sys.path to allow importing P2P_configuration
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# ── Reuse core helpers from P2P ──
from P2P_configuration import (
    EMAIL,
    LOGGER,
    PASSWORD,
    _dismiss_error_dialog,
    check_and_abort_if_locked,
    execute_save_flow,
    get_status_bar_message,
    get_webgui_frame,
    graceful_exit,
    handle_sap_confirmation_dialogs,
    login,
)
# ── Browser / visibility configuration ──
def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}
def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except (TypeError, ValueError):
        return default

SHOW_BROWSER = _env_flag("FIN_SHOW_BROWSER", True)
HEADLESS = not SHOW_BROWSER
BROWSER_CHANNEL = os.getenv("FIN_BROWSER_CHANNEL", "chrome").strip()
SLOW_MO_MS = _env_int("FIN_SLOW_MO_MS", 50 if SHOW_BROWSER else 0)
DEFAULT_BROWSER_ARGS = ["--window-size=1280,800", "--start-maximized"]


async def launch_sap_browser(p, extra_args: list[str] | None = None):
    """Launch Chrome/Chromium in a visible, debug-friendly way."""
    launch_args = list(DEFAULT_BROWSER_ARGS)
    for arg in extra_args or []:
        if arg not in launch_args:
            launch_args.append(arg)

    launch_kwargs = {"headless": HEADLESS, "args": launch_args}
    if SLOW_MO_MS > 0 and not HEADLESS:
        launch_kwargs["slow_mo"] = SLOW_MO_MS

    if BROWSER_CHANNEL:
        try:
            browser = await p.chromium.launch(channel=BROWSER_CHANNEL, **launch_kwargs)
            LOGGER.info(
                f"Browser launched with channel='{BROWSER_CHANNEL}', headless={HEADLESS}, slow_mo={launch_kwargs.get('slow_mo', 0)}ms"
            )
            return browser
        except Exception as e:
            LOGGER.warning(
                f"Could not launch channel '{BROWSER_CHANNEL}' ({e}). Falling back to bundled Chromium."
            )

    browser = await p.chromium.launch(**launch_kwargs)
    LOGGER.info(
        f"Browser launched with bundled Chromium, headless={HEADLESS}, slow_mo={launch_kwargs.get('slow_mo', 0)}ms"
    )
    return browser


async def new_page_in_front(context):
    """Create a page and bring it to the foreground for easier live debugging."""
    page = await context.new_page()
    try:
        await page.bring_to_front()
    except Exception:
        pass
    return page


async def smart_logout(page):
    """Close the SAP session via the logoff service URL.
    Uses the same proven approach as P2P_configuration.py.
    """
    LOGGER.info("SMART LOGOUT -- CLOSING SAP SESSION...")
    try:
        await page.goto(
            "https://my401292.s4hana.cloud.sap/sap/public/bc/icf/logoff?sap-client=100",
            timeout=10000,
        )
        LOGGER.info("SAP SESSION CLOSED SUCCESSFULLY")
    except Exception as e:
        LOGGER.warning(f"LOGOUT NAVIGATION FAILED (IGNORED): {e}")
    return True


JS_FIND_ROW_BY_COCD = """
(targetCoCd) => {
    const targetLower = targetCoCd.trim().toLowerCase();

    // Strategy 1: scan all visible span/td/div elements for exact CoCd text match
    const allVisible = document.querySelectorAll('span, div, input, td');
    for (const el of allVisible) {
        const txt = (el.textContent || '').trim().toLowerCase();
        if (txt !== targetLower) continue;

        // Walk up to find the cell with an M0-style table ID
        let curr = el;
        while (curr && curr !== document.body) {
            if (curr.id && curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/)) {
                const m = curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                return { prefix: m[1], rowIdx: m[2], colIdx: m[3], id: curr.id };
            }
            curr = curr.parentElement;
        }
    }

    // Strategy 2: lsdata search
    const lsEls = document.querySelectorAll('[lsdata]');
    for (const el of lsEls) {
        try {
            const raw = el.getAttribute('lsdata');
            const d = JSON.parse(raw.replace(/'/g, '"'));
            for (let k in d) {
                if (d[k] && typeof d[k] === 'string') {
                    const valLower = d[k].trim().toLowerCase();
                    if (valLower === targetLower || (parseInt(valLower, 10) === parseInt(targetLower, 10) && !isNaN(parseInt(targetLower, 10)))) {
                        let curr = el;
                        while (curr && curr !== document.body) {
                            if (curr.id && curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/)) {
                                const m = curr.id.match(/(M0:\\d+:\\d+)\\[(\\d+),(\\d+)\\]/);
                                return { prefix: m[1], rowIdx: m[2], colIdx: m[3], id: curr.id };
                            }
                            curr = curr.parentElement;
                        }
                    }
                }
            }
        } catch(e) { continue; }
    }
    return null;
}
"""
JS_GET_CHECKBOX_STATE = """
(cellId) => {
    const el = document.getElementById(cellId);
    if (!el) return null;

    // Check for lsdata attribute (SAP stores checked state as '1': true)
    const raw = el.getAttribute('lsdata');
    if (raw) {
        try {
            const d = JSON.parse(raw);
            if (typeof d['1'] === 'boolean') return d['1'];
        } catch(e) {}
    }

    // Check the _c suffix variant
    const cEl = document.getElementById(cellId + '_c');
    if (cEl) {
        const cRaw = cEl.getAttribute('lsdata');
        if (cRaw) {
            try {
                const d = JSON.parse(cRaw);
                if (typeof d['1'] === 'boolean') return d['1'];
            } catch(e) {}
        }
    }

    // Check for native input checkbox inside
    const inp = el.querySelector('input[type="checkbox"]');
    if (inp) return inp.checked;

    // Check aria-checked
    if (el.getAttribute('aria-checked')) return el.getAttribute('aria-checked') === 'true';

    return null;
}
"""


async def company_code_gl_view_106039(targets: list[dict]):
    """
    Company Code General Ledger View (SSCUI 106039)
    ────────────────────────────────────────────────
    Automates editing of company code GL settings on the list view.

    Target format:
        {
            "CoCd": "1810",
            "Max_ex_dev": "10",                          # Max exchange rate deviation (text)
            "No_Exch_Rate_Diff": False,                   # checkbox: No Exch. Rate Diff. When Clearing
            "Negative_Postings_Permitted": True,           # checkbox: Negative Postings Permitted
            "Enable_Amount_Split": False                   # checkbox: Enable Amount Split
        }

    Column layout (0-indexed from the SAP table):
        Col 1 = CoCd (read-only identifier)
        Col 2 = Company Name (read-only)
        Col 3 = Max.ex.dev (editable text)
        Col 4 = No Exch. Rate Diff. When Clearin... (checkbox)
        Col 5 = Negative Postings Permitted (checkbox)
        Col 6 = Enable Amount Split (checkbox)

    NOTE: Checkbox columns may be offset by ±1 depending on row-selector column.
          The JS dynamically detects the correct column index.
    """
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=V_001_CLD_GL&CustomizingObject=V_001_CLD_GL&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ER9_11002557&Type=SSCUI"
    )

    async with async_playwright() as p:
        browser = await launch_sap_browser(p, extra_args=["--disable-gpu"])
        page = await new_page_in_front(context)

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            LOGGER.info("CHECKING FOR LOCKED DATA WARNING...")
            if await check_and_abort_if_locked(page):
                return

            await asyncio.sleep(5)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            LOGGER.info(f"WEBGUI FRAME FOUND. Processing {len(targets)} target(s)...")

            for idx, target in enumerate(targets, start=1):
                cocd = str(target.get("CoCd", "")).strip()
                max_ex_dev = target.get("Max_ex_dev")
                no_exch = target.get("No_Exch_Rate_Diff")
                neg_post = target.get("Negative_Postings_Permitted")
                amt_split = target.get("Enable_Amount_Split")

                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING CoCd={cocd}")

                # ── Step 1: Find the row ──
                row_info = None
                for scroll_attempt in range(12):
                    try:
                        # Try Position... button on first attempt
                        if scroll_attempt == 0:
                            try:
                                pos_btn = webgui_frame.locator("text=/Position/i").first
                                if await pos_btn.is_visible(timeout=1000):
                                    await pos_btn.click(force=True)
                                    await asyncio.sleep(0.5)
                                    await page.keyboard.type(cocd, delay=0)
                                    await page.keyboard.press("Enter")
                                    LOGGER.info(f"Used Position... to jump to '{cocd}'")
                                    await asyncio.sleep(1)
                            except Exception:
                                pass

                        row_info = await webgui_frame.evaluate(
                            JS_FIND_ROW_BY_COCD, cocd
                        )
                        if row_info:
                            LOGGER.info(
                                f"Found CoCd '{cocd}' at row={row_info['rowIdx']}, prefix={row_info['prefix']}"
                            )
                            break

                        LOGGER.info(
                            f"CoCd '{cocd}' not visible. PageDown ({scroll_attempt + 1}/12)..."
                        )
                        await webgui_frame.locator("body").click(force=True)
                        await page.keyboard.press("PageDown")
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        LOGGER.error(f"Error finding row: {e}")
                        await asyncio.sleep(1)

                if not row_info:
                    LOGGER.error(f"CoCd '{cocd}' NOT FOUND after scrolling. Skipping.")
                    continue

                prefix = row_info["prefix"]
                row_idx = row_info["rowIdx"]

                # ── Step 2: Detect column layout ──
                # The CoCd column index tells us the offset.
                # CoCd is typically col 2 (if col 1 is row-selector checkbox).
                # Max.ex.dev is CoCd_col + 2, checkboxes follow after.
                cocd_col = int(row_info["colIdx"])
                max_col = cocd_col + 2  # Max.ex.dev
                chk_no_exch_col = cocd_col + 3  # No Exch. Rate Diff.
                chk_neg_post_col = cocd_col + 4  # Negative Postings Permitted
                chk_amt_split_col = cocd_col + 5  # Enable Amount Split

                LOGGER.info(
                    f"  Column layout: CoCd={cocd_col}, Max={max_col}, NoExch={chk_no_exch_col}, NegPost={chk_neg_post_col}, AmtSplit={chk_amt_split_col}"
                )

                # ── Step 3: Edit Max.ex.dev (text field) ──
                if max_ex_dev is not None:
                    max_cell_id = f"{prefix}[{row_idx},{max_col}]_c"
                    LOGGER.info(
                        f"  Setting Max.ex.dev to '{max_ex_dev}' (cell: {max_cell_id})"
                    )

                    # Scroll into view
                    await webgui_frame.evaluate(f"""() => {{
                        const el = document.getElementById('{max_cell_id}');
                        if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center', inline: 'center' }});
                    }}""")
                    await asyncio.sleep(0.3)

                    inp = webgui_frame.locator(f"id={max_cell_id}")
                    if await inp.count() > 0:
                        try:
                            await inp.first.click(force=True, timeout=2000)
                        except Exception:
                            await webgui_frame.evaluate(
                                f"() => document.getElementById('{max_cell_id}')?.click()"
                            )
                        await asyncio.sleep(0.3)

                        await page.keyboard.press("F2")
                        await asyncio.sleep(0.2)

                        # Force clear + type
                        await webgui_frame.evaluate(f"""() => {{
                            let cell = document.getElementById('{max_cell_id}');
                            if (!cell) return;
                            let inp = (cell.tagName === 'INPUT' || cell.tagName === 'TEXTAREA') ? cell : cell.querySelector('input, textarea');
                            if (inp) {{ inp.focus(); inp.value = ''; inp.dispatchEvent(new Event('input', {{ bubbles: true }})); }}
                        }}""")
                        await asyncio.sleep(0.2)

                        await page.keyboard.type(str(max_ex_dev), delay=0)
                        await asyncio.sleep(0.1)
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.2)
                        LOGGER.info(f"  ✔ Max.ex.dev set to '{max_ex_dev}'")
                    else:
                        LOGGER.warning(f"  Max.ex.dev input not found at {max_cell_id}")

                # ── Step 4: Toggle checkboxes ──
                checkbox_tasks = [
                    (chk_no_exch_col, no_exch, "No_Exch_Rate_Diff"),
                    (chk_neg_post_col, neg_post, "Negative_Postings_Permitted"),
                    (chk_amt_split_col, amt_split, "Enable_Amount_Split"),
                ]
                for col, desired_state, name in checkbox_tasks:
                    if desired_state is None:
                        continue  # Not specified, skip
                    # Try both ID patterns: with _c suffix and without
                    cell_id_c = f"{prefix}[{row_idx},{col}]_c"
                    cell_id = f"{prefix}[{row_idx},{col}]"
                    # Read current state
                    current = await webgui_frame.evaluate(
                        JS_GET_CHECKBOX_STATE, cell_id
                    )
                    if current is None:
                        current = await webgui_frame.evaluate(
                            JS_GET_CHECKBOX_STATE, cell_id_c
                        )
                    LOGGER.info(f"  {name}: current={current}, desired={desired_state}")
                    if current == desired_state:
                        LOGGER.info(f"  ✔ {name} already correct. Skipping.")
                        continue
                    # Need to toggle — click the checkbox element
                    LOGGER.info(f"  Toggling {name}...")
                    toggled = False
                    for try_id in [cell_id_c, cell_id]:
                        el = webgui_frame.locator(f"id={try_id}")
                        if await el.count() > 0:
                            try:
                                await el.first.click(force=True, timeout=2000)
                                toggled = True
                                LOGGER.info(f"  ✔ {name} toggled via {try_id}")
                                await asyncio.sleep(0.5)
                                break
                            except Exception:
                                continue
                    if not toggled:
                        # JS fallback click
                        clicked = await webgui_frame.evaluate(f"""() => {{
                            const el = document.getElementById('{cell_id_c}') || document.getElementById('{cell_id}');
                            if (el) {{ el.click(); return true; }}
                            return false;
                        }}""")
                        if clicked:
                            LOGGER.info(f"  ✔ {name} toggled via JS fallback")
                            await asyncio.sleep(0.5)
                        else:
                            LOGGER.error(f"  ✘ Could not toggle {name}")
                LOGGER.info(f"  ✔ CoCd '{cocd}' — all fields processed.")
            # ── Step 5: Save ──
            LOGGER.info("SAVING ALL CHANGES...")
            saved = await execute_save_flow(page)
            if saved:
                LOGGER.info("DONE. INITIATING GRACEFUL EXIT...")
                await smart_logout(page)
            else:
                LOGGER.warning("Save flow did not confirm. Check manually.")

        except Exception as e:
            LOGGER.error(f"FATAL ERROR: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


async def create_global_company_106040(targets: list[dict]):
    """
    Create Company (SSCUI 106040 / V_880_CLD)
    Edits Company Name and Country in the list view.
    """
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=V_880_CLD&CustomizingObject=V_880_CLD&"
        "CustomizingObjectType=V&CustomizingProject=&"
        "CustomizingTransaction=S_ER9_11002556&Type=SSCUI"
    )
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        page = await new_page_in_front(context)

        async def _local_edit(wf, cell_id, value):
            """Click cell, select ALL text inside the input via JS, then type to replace."""
            try:
                # Scroll into view
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{cell_id}');
                    if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                }}""")
                await asyncio.sleep(0.3)
                loc = wf.locator(f"id={cell_id}")
                if await loc.count() == 0:
                    return False
                # SINGLE click (no force, no double-click) to activate the cell
                await loc.first.click(timeout=2000)
                await asyncio.sleep(0.4)

                # Use JS to find the ACTUAL input element and select ALL text inside it
                # This is the key fix: setSelectionRange works on text INSIDE the input,
                # unlike Home/End/Ctrl+A which navigate TABLE CELLS in SAP WebGUI
                selected = await wf.evaluate(f"""() => {{
                    // Try the cell itself, or find an input inside it
                    let el = document.getElementById('{cell_id}');
                    if (!el) return false;
                    let inp = el;
                    if (el.tagName !== 'INPUT' && el.tagName !== 'TEXTAREA') {{
                        inp = el.querySelector('input, textarea');
                    }}
                    // Fallback: use the currently active/focused element
                    if (!inp || (inp.tagName !== 'INPUT' && inp.tagName !== 'TEXTAREA')) {{
                        inp = document.activeElement;
                    }}
                    if (inp && (inp.tagName === 'INPUT' || inp.tagName === 'TEXTAREA')) {{
                        inp.focus();
                        inp.setSelectionRange(0, inp.value.length);
                        return true;
                    }}
                    return false;
                }}""")

                if not selected:
                    LOGGER.warning(f"  Could not select text in {cell_id}")
                    return False

                await asyncio.sleep(0.1)

                # Type the new value — replaces the selected text
                await page.keyboard.type(str(value), delay=30)
                await asyncio.sleep(0.2)

                # Tab to confirm and move to next cell
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.3)
                return True
            except Exception as e:
                LOGGER.warning(f"  _local_edit failed for {cell_id}: {e}")
                return False

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(2)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                return

            to_create = []

            for idx, target in enumerate(targets, start=1):
                company_id = str(target.get("Company", "")).strip()
                name = target.get("Name")
                country = target.get("Country")

                LOGGER.info(f"[{idx}/{len(targets)}] Company={company_id}")

                # Find row
                row_info = None
                for scroll_attempt in range(5):  # Reduced from 10
                    if scroll_attempt == 0:
                        try:
                            pos_btn = webgui_frame.locator("text=/Position/i").first
                            if await pos_btn.is_visible(timeout=1000):
                                await pos_btn.click(force=True)
                                await asyncio.sleep(0.5)
                                await page.keyboard.type(company_id, delay=0)  # delay=0
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(1.0)  # Reduced from 1.5
                        except:
                            pass

                    row_info = await webgui_frame.evaluate(
                        JS_FIND_ROW_BY_COCD, company_id
                    )
                    if row_info:
                        break
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(0.5)  # Reduced from 1.0

                if not row_info:
                    LOGGER.info(
                        f"  Company '{company_id}' not found. Queuing for New Entries."
                    )
                    to_create.append(target)
                    continue

                prefix = row_info["prefix"]
                row_idx = row_info["rowIdx"]
                col_idx = int(row_info["colIdx"])

                # Edit Name of Company (Col + 1)
                if name:
                    cell = f"{prefix}[{row_idx},{col_idx + 1}]"
                    if not await _local_edit(webgui_frame, cell, name):
                        await _local_edit(webgui_frame, cell + "_c", name)
                    LOGGER.info(f"  ✔ Name → '{name}'")

                # Edit Country/Region (Col + 2)
                if country:
                    cell = f"{prefix}[{row_idx},{col_idx + 2}]"
                    if not await _local_edit(webgui_frame, cell, country):
                        await _local_edit(webgui_frame, cell + "_c", country)
                    LOGGER.info(f"  ✔ Country → '{country}'")

                LOGGER.info(f"  ✔ Company '{company_id}' updated.")

            # Batch create new entries
            if to_create:
                LOGGER.info(f"Creating {len(to_create)} new entries...")
                new_clicked = False
                for frame in page.frames:
                    try:
                        btn = frame.locator("text=/New Entries/i").first
                        if await btn.is_visible(timeout=2000):
                            await btn.click(force=True)
                            new_clicked = True
                            LOGGER.info("  ✔ 'New Entries' clicked.")
                            await asyncio.sleep(3)
                            break
                    except:
                        continue

                if new_clicked:
                    webgui_frame = await get_webgui_frame(page) or webgui_frame
                    for cidx, target in enumerate(to_create, start=1):
                        cid = str(target.get("Company", "")).strip()
                        nm = target.get("Name", "")
                        cnt = target.get("Country", "")

                        # Anchor to row cidx col 0
                        # Usually row index starts at 1 or depends on header.
                        # We'll use the purchasing_group strategy of finding the row suffix.
                        cell_clicked = await webgui_frame.evaluate(
                            f"""(rowIdx) => {{
                            const suffix1 = '[' + rowIdx + ',1]';
                            const suffix2 = '[' + rowIdx + ',1]_c';
                            const elements = document.querySelectorAll('[id*="["]');
                            for (const el of elements) {{
                                if (el.id.endsWith(suffix1) || el.id.endsWith(suffix2)) {{
                                    el.scrollIntoView({{ behavior: 'instant', block: 'center', inline: 'start' }});
                                    el.click();
                                    return el.id;
                                }}
                            }}
                            return null;
                        }}""",
                            cidx,
                        )  # cidx starts at 1, which matches the first data row in most SAP tables

                        if not cell_clicked:
                            await page.keyboard.press("Home")
                            await asyncio.sleep(0.2)

                        # Type fields: Company, Name, Country
                        # Use Tab to navigate
                        await page.keyboard.type(cid, delay=0)
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.1)
                        await page.keyboard.type(nm, delay=0)
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.1)
                        await page.keyboard.type(cnt, delay=0)
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.1)
                        LOGGER.info(f"  ✔ New entry '{cid}' filled.")

            await execute_save_flow(page)
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"Task 2 Error: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


async def wait_for_ui_stable(page, seconds=5):
    """Wait for UI to stabilize."""
    await asyncio.sleep(seconds)


async def handle_busy_indicators(frame, timeout_ms=10000):
    """Waits for SAP busy indicators to disappear."""
    try:
        page = frame.page if hasattr(frame, "page") else frame
        busy = page.locator(
            ".sapUiLocalBusyIndicator, .sapUiBlockLayer, [title='Please wait']"
        ).filter(visible=True)
        count = await busy.count()
        if count > 0:
            LOGGER.info(f"⏳ Waiting for {count} busy indicators...")
            for i in range(count):
                await busy.nth(i).wait_for(state="hidden", timeout=timeout_ms)
            await asyncio.sleep(1)
    except:
        pass


async def handle_sap_warning(frame):
    """Clicks 'Yes' on warning dialogs if they appear."""
    try:
        page = frame.page if hasattr(frame, "page") else frame
        warning_dialog = (
            page.locator(".sapMDialog")
            .filter(has_text=re.compile("Warning|discarded", re.I))
            .filter(visible=True)
            .first
        )
        if await warning_dialog.is_visible(timeout=2000):
            LOGGER.info("⚠️ Handling Warning dialog...")
            yes_btn = warning_dialog.locator("button:has-text('Yes')").first
            await yes_btn.click()
            await asyncio.sleep(2)
            return True
    except:
        pass
    return False


async def handle_sap_information_dialog(frame):
    """Clicks 'Close' on information dialogs if they appear."""
    try:
        page = frame.page if hasattr(frame, "page") else frame
        info_dialog = (
            page.locator(".sapMDialog")
            .filter(has_text=re.compile("Information|Already Assigned", re.I))
            .filter(visible=True)
            .first
        )
        if await info_dialog.is_visible(timeout=3000):
            LOGGER.info("ℹ️ Handling Information dialog...")
            close_btn = info_dialog.locator(
                "button:has-text('Close'), button:has-text('OK')"
            ).first
            await close_btn.click()
            await asyncio.sleep(2)
            return True
    except:
        pass
    return False


async def ensure_wizard_step_1(frame):
    """Ensures we are at Step 1 of the wizard. If not, clicks Previous or Step 1 header."""
    try:
        page = frame.page if hasattr(frame, "page") else frame
        # Check if Step 1 is active
        step1_active = await page.locator(
            ".sapMWizardProgressNavStep[aria-label*='Step 1'][data-sap-ui-wpn-step-open='true']"
        ).is_visible(timeout=2000)
        if step1_active:
            return True

        LOGGER.info("🔄 Resetting wizard to Step 1...")
        # Try clicking Step 1 header directly
        step1_header = page.locator(
            ".sapMWizardProgressNavStep[aria-label*='Step 1']"
        ).first
        if await step1_header.is_visible():
            await step1_header.click()
            await asyncio.sleep(2)
            await handle_sap_warning(frame)
            return True

        # Fallback: click Previous until Step 1
        for _ in range(3):
            prev_btn = (
                page.locator("button:has-text('Previous')").filter(visible=True).first
            )
            if await prev_btn.is_visible():
                await prev_btn.click()
                await asyncio.sleep(2)
                await handle_sap_warning(frame)
                if await page.locator("button:has-text('Step 2')").is_visible():
                    return True
    except Exception as e:
        LOGGER.warning(f"Wizard reset failed: {e}")
    return False


async def select_ui5_dropdown(frame, index, value):
    """Robust logic to select a visible combobox by its index."""
    try:
        LOGGER.info(f"🔽 Selecting index {index} → {value}")
        page = frame.page if hasattr(frame, "page") else frame

        # 1. Get visible comboboxes
        combos = page.locator("[role='combobox']").filter(visible=True)
        await combos.first.wait_for(timeout=5000)

        if await combos.count() <= index:
            LOGGER.warning(
                f"❌ Index {index} out of range (Found {await combos.count()} visible combos)"
            )
            return False

        combo = combos.nth(index)

        # 1b. Smart skip
        current_val = ""
        try:
            inner = combo.locator("span.sapMSltLabel, .sapMSltArrow ~ span").first
            current_val = (
                await inner.text_content()
                if await inner.count() > 0
                else await combo.text_content()
            )
        except:
            pass

        if current_val and value.lower() in current_val.lower():
            LOGGER.info(f"✨ '{value}' already selected. Skipping.")
            return True

        # 2. Click and select
        await combo.click(force=True)
        await asyncio.sleep(2)

        option = (
            page.locator(
                f"[role='option']:has-text('{value}'), [role='listitem']:has-text('{value}'), .sapMSelectListItemBase:has-text('{value}')"
            )
            .filter(visible=True)
            .first
        )
        if not await option.is_visible(timeout=3000):
            await combo.press("ArrowDown")  # Fallback open
            option = (
                page.locator(f"[role='option'], .sapMSelectListItemBase")
                .filter(has_text=re.compile(f"^{value}", re.I))
                .filter(visible=True)
                .first
            )

        await option.wait_for(timeout=5000)
        await option.click(force=True)
        await handle_busy_indicators(frame)
        return True
    except Exception as e:
        LOGGER.error(f"❌ Dropdown failed: {e}")
        return False


async def select_from_value_help(frame, label_text, target_value):
    """Fiori Value Help: Search + Go + Select."""
    try:
        LOGGER.info(f"🔍 Value Help → {label_text}: {target_value}")
        page = frame.page if hasattr(frame, "page") else frame

        icon = page.locator(
            f"[aria-label*='{label_text}'] .sapMInputBaseIcon, label:has-text('{label_text}') >> xpath=following::*[contains(@class,'sapMInputBaseIcon')][1]"
        ).first
        await icon.click(force=True)
        await asyncio.sleep(3)
        await handle_busy_indicators(page)

        # Search field
        search = page.locator(
            ".sapMDialog input[type='search'], .sapMDialog input[placeholder='Search']"
        ).first
        await search.fill(target_value)
        await search.press("Enter")
        await asyncio.sleep(2)
        await handle_busy_indicators(page)

        # Select result
        cell = (
            page.locator(
                f".sapMDialog td:has-text('{target_value}'), .sapMDialog .sapMLIBContent:has-text('{target_value}')"
            )
            .filter(visible=True)
            .first
        )
        await cell.click()
        await asyncio.sleep(1)

        # Confirm if OK button exists
        ok_btn = (
            page.locator(
                ".sapMDialog button:has-text('OK'), .sapMDialog button:has-text('Select')"
            )
            .filter(visible=True)
            .first
        )
        if await ok_btn.is_visible():
            await ok_btn.click()
        else:
            await cell.dblclick()

        await handle_busy_indicators(page)
        return True
    except Exception as e:
        LOGGER.error(f"❌ Value Help failed: {e}")
        return False


async def execute_financial_accounting_automation_100297(targets):
    """TOTAL CODE: Main automation function for SAP Financial Accounting."""
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        LOGGER.info(f"🚀 Navigating to Home...")
        await page.goto(
            "https://my401292.s4hana.cloud.sap/ui#Shell-home", timeout=60000
        )

        # Login
        await login(page, EMAIL, PASSWORD)
        await asyncio.sleep(8)

        # 1. Search in Shell Header
        LOGGER.info("🔍 Searching for 100297 in Shell Header...")
        try:
            search_trigger = page.locator("#sf").first
            await search_trigger.click(timeout=10000)
            await page.keyboard.type("100297")
            await page.keyboard.press("Enter")
            await asyncio.sleep(6)

            # Click on 'Configure Your Solution' result
            result = page.locator(
                ".sapUshellSearchResultListItem:has-text('Configure Your Solution')"
            ).first
            await result.click(timeout=10000)
            await asyncio.sleep(8)

            # 2. Filter inside the app
            LOGGER.info("🔍 Filtering inside 'Configure Your Solution'...")
            inner_search = (
                page.locator("input[type='search'], .sapMSearchFieldI")
                .filter(visible=True)
                .first
            )
            await inner_search.fill("100297", timeout=10000)
            await inner_search.press("Enter")
            await asyncio.sleep(3)

            # 3. Click Configure
            LOGGER.info("🖱 Clicking 'Configure'...")
            conf_btn = (
                page.locator("button:has-text('Configure')").filter(visible=True).first
            )
            await conf_btn.click()
            await asyncio.sleep(8)
            await handle_busy_indicators(page)
        except Exception as ne:
            LOGGER.error(f"❌ Navigation failed: {ne}")
            # Fallback: try direct URL if known
            pass

        for target in targets:
            area = target["area"]
            subarea = target["subarea"]
            process = target["process"]
            coa = target["chart_of_accounts"]
            tkeys = target.get("transaction_keys", [])

            LOGGER.info(f"🎯 STARTING: {process}")
            await ensure_wizard_step_1(page)
            await asyncio.sleep(2)

            # Step 1: Area/Subarea/Process
            await select_ui5_dropdown(page, 0, area)
            await handle_sap_warning(page)
            await select_ui5_dropdown(page, 1, subarea)
            await handle_sap_warning(page)
            await select_ui5_dropdown(page, 2, process)
            await handle_sap_warning(page)

            # Transition to Step 2
            step2_btn = (
                page.locator("button:has-text('Step 2')").filter(visible=True).first
            )
            await step2_btn.click()
            await wait_for_ui_stable(page, 3)

            # Step 2: Parameters (Transaction Key is index 3 as it is the 4th combo)
            for tk in tkeys:
                LOGGER.info(f"⚙️ Parameter: {tk}")
                await select_ui5_dropdown(page, 3, tk)
                await select_from_value_help(page, "Chart of Accounts", coa)

                # Transition to Step 3
                LOGGER.info("➡️ Moving to Step 3...")
                step3_btn = (
                    page.locator(
                        "[aria-label*='Step 3'], button:has-text('Step 3'), .sapMWizardStep:has-text('Step 3')"
                    )
                    .filter(visible=True)
                    .first
                )
                await step3_btn.click(force=True)
                await asyncio.sleep(3)

                # Handle "Information" popup if it appears
                await handle_sap_information_dialog(page)
                await wait_for_ui_stable(page, 3)

                # Step 3: Account Assignments


# async def Maintain_Budget_Availability_Control_Profile_for_Cost_Centers_102781(
#     targets: list[dict],
# ):
#     """
#     Maintain Budget Availability Control Profile for Cost Centers (SSCUI 102781)
#     ───────────────────────────────────────────────────────────────────────────
#     Logic:
#       1. Use 'Position' to jump to each profile.
#       2. If found, update in-place immediately.
#       3. If not found, queue for 'New Entries' pass.
#       4. After 'New Entries', MUST navigate to 'Account Groups' sub-folder
#          to assign a default group (e.g. 'ALL') or SAP will refuse to save.
#       5. Save and handle post-save dialogs.
#     """
#     url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FINSVC_AVC_PROF_CC&CustomizingObject=FINSVC_AVC_PROF_CC&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ER9_11001078&Type=SSCUI"

#     async with async_playwright() as p:
#         browser = await launch_sap_browser(p)
#         context = await browser.new_context()
#         page = await new_page_in_front(context)

#         try:
#             LOGGER.info(f"OPENING: {url}")
#             await page.goto(url, wait_until="commit", timeout=60000)
#             await login(page, EMAIL, PASSWORD)

#             if await check_and_abort_if_locked(page):
#                 return

#             webgui_frame = await get_webgui_frame(page)
#             if not webgui_frame:
#                 LOGGER.error("WEBGUI IFRAME NOT FOUND!")
#                 return

#             LOGGER.info(f"WEBGUI FRAME FOUND. Processing {len(targets)} target(s)...")
#             new_entries_queue: list[dict] = []

#             # ── PASS 1: SEARCH & UPDATE EXISTING ──────────────────────────
#             for idx, target in enumerate(targets, start=1):
#                 p_id = str(target.get("Profile", "")).strip()
#                 LOGGER.info(f"[{idx}/{len(targets)}] Searching for '{p_id}'...")

#                 # Position jump
#                 try:
#                     pos_btn = webgui_frame.locator("text=/Position/i").first
#                     if await pos_btn.is_visible(timeout=3000):
#                         await pos_btn.click(force=True)
#                         await asyncio.sleep(1)
#                         await page.keyboard.type(p_id)
#                         await page.keyboard.press("Enter")
#                         await asyncio.sleep(1)
#                 except:
#                     pass

#                 # Check if row exists
#                 row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, p_id)
#                 if not row_info:
#                     LOGGER.info(f"  '{p_id}' NOT found → queued for New Entries.")
#                     new_entries_queue.append(target)
#                     continue

#                 # Update logic (skipped for brevity here, assuming focus on New Entries fix)
#                 LOGGER.info(
#                     f"  ✔ Found '{p_id}' at row {row_info['rowIdx']}. Updating fields..."
#                 )

#             # ── PASS 2: BULK CREATE NEW ──────────────────────────────────
#             if new_entries_queue:
#                 LOGGER.info(f"Creating {len(new_entries_queue)} new entry/entries...")

#                 # 1. Click New Entries
#                 for frame in page.frames:
#                     btn = frame.locator("text=/New Entries/i").first
#                     if await btn.is_visible(timeout=3000):
#                         await btn.click(force=True)
#                         await asyncio.sleep(4)
#                         break

#                 webgui_frame = await get_webgui_frame(page) or webgui_frame

#                 for cidx, target in enumerate(new_entries_queue, start=1):
#                     p_id = str(target.get("Profile", "")).strip()
#                     ctrl = str(target.get("Availy Ctrl Type", ""))
#                     name = str(target.get("Availy Prfl Name", ""))
#                     trange = str(
#                         target.get("Time_Range") or target.get("Time Range", "")
#                     )
#                     b_curr = str(target.get("Budget Currency Type", ""))

#                     LOGGER.info(f"  [NEW {cidx}] Filling Header for '{p_id}'...")

#                     # Focus first cell of row
#                     await page.keyboard.press("Home")
#                     await asyncio.sleep(0.2)

#                     # Fill row fields
#                     vals = [p_id, ctrl, name, trange, b_curr]
#                     for val in vals:
#                         await page.keyboard.down("Control")
#                         await page.keyboard.press("a")
#                         await page.keyboard.up("Control")
#                         await page.keyboard.press("Delete")
#                         if val:
#                             await page.keyboard.type(str(val))
#                         await page.keyboard.press("Tab")
#                         await asyncio.sleep(0.1)

#                 # 2. SELECT THE ROW and GO TO SUB-VIEW (Account Groups)
#                 # This is CRITICAL. SAP won't save without an account group.
#                 LOGGER.info(
#                     "  ↪ Navigating to 'Account Groups' sub-view to finalize profile(s)..."
#                 )

#                 folder_found = False
#                 for sel in [
#                     "text='Account Groups'",
#                     "text=/Account Groups/i",
#                     "[title*='Account Groups']",
#                 ]:
#                     folder = webgui_frame.locator(sel).first
#                     if await folder.is_visible(timeout=3000):
#                         await folder.click(
#                             click_count=2, force=True
#                         )  # Double-click to enter
#                         LOGGER.info(
#                             f"  ✔ Entered 'Account Groups' sub-view via '{sel}'"
#                         )
#                         folder_found = True
#                         break

#                 if folder_found:
#                     await asyncio.sleep(3)
#                     # Click New Entries in the sub-view
#                     sub_new = webgui_frame.locator("text=/New Entries/i").first
#                     if await sub_new.is_visible(timeout=3000):
#                         await sub_new.click(force=True)
#                         await asyncio.sleep(2)

#                         LOGGER.info("  ✔ Adding 'ALL' to Account Groups...")
#                         await page.keyboard.type("ALL")
#                         await page.keyboard.press("Enter")
#                         await asyncio.sleep(1)
#                 else:
#                     LOGGER.error(
#                         "  ✘ Could not find 'Account Groups' folder in left pane!"
#                     )

#             # ── PASS 3: SAVE & DIALOGS ────────────────────────────────────
#             LOGGER.info("SAVING ALL CHANGES...")
#             await execute_save_flow(page)

#             for dialog_num in range(1, 3):
#                 await asyncio.sleep(3)
#                 LOGGER.info(f"Checking for post-save dialog #{dialog_num}...")

#                 dialog_found = False
#                 for frame in page.frames:
#                     try:
#                         for sel in [
#                             'button[title*="Continue"]',
#                             'button[title*="OK"]',
#                             '[title*="Continue (Enter)"]',
#                             "#btn\\[0\\]",
#                             '.sapMBtn:has-text("OK")',
#                             'button[aria-label*="Continue"]',
#                         ]:
#                             btn = frame.locator(sel).first
#                             if await btn.is_visible(timeout=1000):
#                                 await btn.click(force=True)
#                                 LOGGER.info(
#                                     f"  ✔ Dialog dismissed via '{sel}' (Green Tick)"
#                                 )
#                                 dialog_found = True
#                                 break
#                         if dialog_found:
#                             break
#                     except:
#                         continue

#                 if not dialog_found:
#                     LOGGER.info(
#                         f"  Dialog #{dialog_num} not found via specific selectors, sending Enter..."
#                     )
#                     await page.keyboard.press("Enter")
#                 await asyncio.sleep(2)


#         except Exception as e:
#             LOGGER.error(f"FATAL ERROR in 102781: {e}", exc_info=True)
#         finally:
#             if not page.is_closed():
#                 await smart_logout(page)
#                 await browser.close()
async def Maintain_Budget_Availability_Control_Profile_for_Cost_Centers_102781(
    targets: list[dict],
):
    import asyncio

    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FINSVC_AVC_PROF_CC&CustomizingObject=FINSVC_AVC_PROF_CC&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ER9_11001078&Type=SSCUI"
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context()
        page = await new_page_in_front(context)

        async def sap_wait(t=1):
            await asyncio.sleep(t)

        async def refresh_frame():
            frame = await get_webgui_frame(page)
            if not frame:
                raise Exception("WEBGUI FRAME NOT FOUND")
            return frame

        async def focus_first_input():
            sels = ["input", '[role="textbox"]', 'td[tabindex="0"]', ".urTxtStd"]
            for sel in sels:
                try:
                    obj = webgui_frame.locator(sel).first
                    if await obj.is_visible(timeout=1500):
                        await obj.click(force=True)
                        await sap_wait(0.2)
                        return
                except:
                    pass
            raise Exception("NO EDITABLE INPUT FOUND")

        async def clear_field():
            await page.keyboard.down("Control")
            await page.keyboard.press("A")
            await page.keyboard.up("Control")
            await page.keyboard.press("Delete")
            await sap_wait(0.1)

        async def type_fast(val):
            await clear_field()
            if str(val).strip():
                await page.keyboard.type(str(val), delay=15)
            await sap_wait(0.15)

        async def fill_row(profile, ctrl, name, trange, bcurr):
            await focus_first_input()
            vals = [profile, ctrl, name, trange, bcurr]
            for idx, val in enumerate(vals, start=1):
                LOGGER.info(f"FILLING COL {idx}: {val}")
                await type_fast(val)
                await page.keyboard.press("Tab")
                await sap_wait(0.25)

        async def click_new_entries():
            for frame in page.frames:
                for sel in [
                    "text=/New Entries/i",
                    '[title*="New Entries"]',
                    'button:has-text("New Entries")',
                ]:
                    try:
                        btn = frame.locator(sel).first
                        if await btn.is_visible(timeout=1200):
                            await btn.click(force=True)
                            await sap_wait(1.5)
                            return True
                    except:
                        pass
            return False

        async def open_account_groups():
            selectors = [
                "text='Account Groups'",
                "text=/Account Groups/i",
                "[title*='Account Groups']",
            ]
            for sel in selectors:
                try:
                    node = webgui_frame.locator(sel).first
                    if await node.is_visible(timeout=1500):
                        await node.click(force=True)
                        await sap_wait(1)
                        return True
                except:
                    pass
            return False

        async def add_account_group():
            LOGGER.info("ADDING ACCOUNT GROUP")
            selectors = [
                "table input",
                'td[tabindex="0"]',
                ".urTxtStd",
                '[role="textbox"]',
            ]
            focused = False
            for sel in selectors:
                try:
                    objs = webgui_frame.locator(sel)
                    count = await objs.count()
                    if count > 0:
                        target = objs.nth(0)
                        if await target.is_visible(timeout=1200):
                            await target.click(force=True)
                            await sap_wait(0.3)
                            focused = True
                            break
                except:
                    pass
            if not focused:
                raise Exception("ACCOUNT GROUP GRID NOT FOCUSED")
            await page.keyboard.down("Control")
            await page.keyboard.press("A")
            await page.keyboard.up("Control")
            await page.keyboard.press("Delete")
            await sap_wait(0.1)
            await page.keyboard.type("ALL", delay=15)
            await sap_wait(0.3)
            LOGGER.info("ACCOUNT GROUP ADDED")

        async def click_save():
            LOGGER.info("CLICKING SAVE BUTTON")
            selectors = [
                'button:has-text("Save")',
                'text="Save"',
                '[title*="Save"]',
                '.sapMBtn:has-text("Save")',
            ]
            for frame in page.frames:
                for sel in selectors:
                    try:
                        btn = frame.locator(sel).first
                        if await btn.is_visible(timeout=1200):
                            await btn.click(force=True)
                            await sap_wait(1.5)
                            return True
                    except:
                        pass
            try:
                await page.keyboard.press("Control+S")
                await sap_wait(1.5)
                return True
            except:
                pass
            return False

        async def click_green_tick():
            LOGGER.info("CLICKING GREEN TICK")
            selectors = [
                '[title*="Continue"]',
                '[title*="OK"]',
                '[title*="Enter"]',
                "#btn\\[0\\]",
                'button[aria-label*="Continue"]',
                ".sapMBtn",
            ]
            for frame in page.frames:
                for sel in selectors:
                    try:
                        btn = frame.locator(sel).first
                        if await btn.is_visible(timeout=1000):
                            await btn.click(force=True)
                            await sap_wait(0.8)
                            return True
                    except:
                        pass
            try:
                await page.keyboard.press("Enter")
                await sap_wait(0.8)
                return True
            except:
                pass
            return False

        async def capture_save_errors():
            await sap_wait(2)
            errors = []
            selectors = [
                ".sapMDialog",
                '[role="dialog"]',
                ".sapMListTblRow",
                "tr",
                ".sapMText",
                ".urTxtStd",
                ".sapMLabel",
            ]
            full_screen_text = []
            for sel in selectors:
                try:
                    objs = page.locator(sel)
                    count = await objs.count()
                    for i in range(count):
                        try:
                            txt = (await objs.nth(i).inner_text()).strip()
                            if txt:
                                full_screen_text.append(txt)
                                lower = txt.lower()
                                if (
                                    "error" in lower
                                    or "couldn't be saved" in lower
                                    or "add at least" in lower
                                    or "required" in lower
                                    or "invalid" in lower
                                    or "cannot" in lower
                                    or "failed" in lower
                                ):
                                    errors.append(txt)
                        except:
                            pass
                except:
                    pass
            LOGGER.info("FULL SCREEN TEXT:")
            for t in full_screen_text:
                LOGGER.info(t)
            return list(set(errors))

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                return
            webgui_frame = await refresh_frame()
            LOGGER.info(f"WEBGUI FRAME FOUND. Processing {len(targets)} targets...")
            results = []
            for idx, target in enumerate(targets, start=1):
                p_id = str(target.get("Profile", "")).strip()
                ctrl = str(target.get("Availy Ctrl Type", "")).strip()
                name = str(target.get("Availy Prfl Name", "")).strip()
                trange = str(
                    target.get("Time_Range", target.get("Time Range", ""))
                ).strip()
                bcurr = str(target.get("Budget Currency Type", "")).strip()
                LOGGER.info(f"[{idx}] CHECKING PROFILE: {p_id}")
                already_exists = False
                try:
                    pos_btn = webgui_frame.locator("text=/Position/i").first
                    if await pos_btn.is_visible(timeout=1500):
                        await pos_btn.click(force=True)
                        await sap_wait(0.3)
                        await page.keyboard.type(p_id, delay=15)
                        await page.keyboard.press("Enter")
                        await sap_wait(0.8)
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, p_id)
                    if row_info:
                        already_exists = True
                        LOGGER.info(f"PROFILE EXISTS: {p_id}")
                        results.append(
                            {
                                "success": True,
                                "profile": p_id,
                                "message": "PROFILE ALREADY EXISTS",
                            }
                        )
                except Exception as e:
                    LOGGER.error(f"POSITION SEARCH FAILED: {e}")
                if already_exists:
                    continue
                LOGGER.info(f"CREATING PROFILE: {p_id}")
                opened = await click_new_entries()
                if not opened:
                    raise Exception("NEW ENTRIES BUTTON NOT FOUND")
                webgui_frame = await refresh_frame()
                await fill_row(p_id, ctrl, name, trange, bcurr)
                LOGGER.info("OPENING ACCOUNT GROUPS")
                ok = await open_account_groups()
                if not ok:
                    raise Exception("ACCOUNT GROUPS NOT FOUND")
                webgui_frame = await refresh_frame()
                await add_account_group()
                LOGGER.info("CLICKING SAVE")
                saved = await click_save()
                if not saved:
                    raise Exception("SAVE BUTTON NOT FOUND")
                errors = await capture_save_errors()
                if errors:
                    LOGGER.error("SAVE ERRORS DETECTED")
                    for err in errors:
                        LOGGER.error(f"SAP ERROR: {err}")
                    results.append(
                        {"success": False, "profile": p_id, "errors": errors}
                    )
                    try:
                        await click_green_tick()
                    except:
                        pass
                    continue
                LOGGER.info(f"PROFILE CREATED SUCCESSFULLY: {p_id}")
                results.append({"success": True, "profile": p_id, "errors": []})
            LOGGER.info("PROCESS COMPLETED SUCCESSFULLY")
            return results
        except Exception as e:
            LOGGER.error(f"FATAL ERROR in SSCUI 102781: {e}", exc_info=True)
            return [{"success": False, "error": str(e)}]
        finally:
            try:
                if not page.is_closed():
                    await smart_logout(page)
                    await browser.close()
            except:
                pass


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3: Assign Company Code to Company (SSCUI 101631)
# ═══════════════════════════════════════════════════════════════════════════

 
async def assign_company_code_101631(targets: list[dict]):
    """
    Assign company code to company (SSCUI 101631)
    ───────────────────────────────────────────────
    Columns: CoCd | Company Name | City | Company (editable)
 
    For each target, finds the row by CoCd and writes the
    company code directly into the "Company" column.
    If the company code is invalid, SAP shows an error and
    the Save button disappears — we detect that and report it.
    """
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=SIMG_CFMENUSAPCOX16&CustomizingObject=V_001_Y&"
        "CustomizingObjectType=V&CustomizingProject=&"
        "CustomizingTransaction=S_ALR_87007374&Type=SSCUI"
    )
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context()
        page = await new_page_in_front(context)
 
        async def _edit_cell(wf, cell_id, value):
            """Click cell → F2 → JS clear → type → Tab."""
            try:
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{cell_id}');
                    if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                }}""")
                await asyncio.sleep(0.2)
 
                loc = wf.locator(f"id={cell_id}")
                if await loc.count() == 0:
                    return False
 
                await loc.first.click(force=True, timeout=2000)
                await asyncio.sleep(0.3)
 
                await page.keyboard.press("F2")
                await asyncio.sleep(0.3)
 
                # JS: find input, clear its value
                await wf.evaluate(f"""() => {{
                    let el = document.getElementById('{cell_id}');
                    if (!el) return;
                    let inp = (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA')
                        ? el : el.querySelector('input, textarea');
                    if (!inp) inp = document.activeElement;
                    if (inp && (inp.tagName === 'INPUT' || inp.tagName === 'TEXTAREA')) {{
                        inp.focus();
                        inp.value = '';
                        inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                    }}
                }}""")
                await asyncio.sleep(0.1)
 
                await page.keyboard.type(str(value), delay=0)
                await asyncio.sleep(0.1)
 
                await page.keyboard.press("Enter")
                await asyncio.sleep(1.0)
                return True
            except Exception as e:
                LOGGER.warning(f"  _edit_cell failed for {cell_id}: {e}")
                return False
 
        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
 
            if await check_and_abort_if_locked(page): return
            await asyncio.sleep(2)
 
            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame: return
 
            errors = []
 
            for idx, target in enumerate(targets, start=1):
                cocd = str(target.get("CoCd", "")).strip()
                company = str(target.get("Company", "")).strip()
 
                LOGGER.info(f"[{idx}/{len(targets)}] CoCd={cocd} → Company={company}")
 
                # Find row by CoCd
                row_info = None
                for scroll_attempt in range(10):
                    if scroll_attempt == 0:
                        try:
                            pos_btn = webgui_frame.locator("text=/Position/i").first
                            if await pos_btn.is_visible(timeout=1000):
                                await pos_btn.click(force=True)
                                await asyncio.sleep(0.5)
                                await page.keyboard.type(cocd)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(1.5)
                        except: pass
 
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)
                    if row_info: break
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(0.5)
 
                if not row_info:
                    LOGGER.error(f"  CoCd '{cocd}' not found in table.")
                    errors.append(f"CoCd '{cocd}': not found in table")
                    continue
 
                prefix = row_info['prefix']
                row_idx = row_info['rowIdx']
                col_idx = int(row_info['colIdx'])
 
                # Company column = CoCd col + 3
                # Layout: CoCd(col) | Company Name(col+1) | City(col+2) | Company(col+3)
                company_cell = f"{prefix}[{row_idx},{col_idx+3}]"
                ok = await _edit_cell(webgui_frame, company_cell, company)
                if not ok:
                    ok = await _edit_cell(webgui_frame, company_cell + "_c", company)
 
                if ok:
                    # Check for immediate SAP error (e.g. "Entry does not exist in T880")
                    # Wait slightly longer to ensure SAP renders the error message and red outline
                    await asyncio.sleep(1.0)
                    error_text = await webgui_frame.evaluate(f"""() => {{
                        // 1. Check if the text "does not exist" appeared on the screen
                        const bodyText = document.body.innerText.toLowerCase();
                        if (bodyText.includes('does not exist in t880') || (bodyText.includes('does not exist') && bodyText.includes('entry'))) {{
                            // Try to grab just the specific error message bar
                            const errEls = document.querySelectorAll('[class*="sapMsgError"], [class*="urMessageText"], [class*="urMsgBarText"]');
                            for (const el of errEls) {{
                                const t = (el.innerText || el.textContent || '').trim();
                                if (t.toLowerCase().includes('does not exist')) return t;
                            }}
                            return "Entry does not exist in T880 (Check entry)";
                        }}
                       
                        // 2. Check if the company cell got a red outline (error state)
                        const el = document.getElementById('{company_cell}');
                        const el_c = document.getElementById('{company_cell}_c');
                        for (const target of [el, el_c]) {{
                            if (!target) continue;
                            if (target.classList.contains('urErr') ||
                                target.classList.contains('lsField--error') ||
                                target.closest('.urErr') ||
                                target.closest('.lsField--error') ||
                                (target.getAttribute('title') || '').toLowerCase().includes('error')) {{
                                return "Invalid entry correctly rejected by SAP (red outline)";
                            }}
                        }}
                        return null;
                    }}""")
 
                    if error_text:
                        LOGGER.error(f"  ✘ SAP ERROR: {error_text}. Aborting save!")
                        errors.append(f"CoCd '{cocd}' → Company '{company}': {error_text}")
                        break  # Stop processing further targets and abort
                    else:
                        LOGGER.info(f"  ✔ Company set to '{company}'")
                else:
                    LOGGER.error(f"  ✘ Could not edit Company cell for CoCd '{cocd}'")
                    errors.append(f"CoCd '{cocd}': could not edit cell")
 
            # Report any errors before save
            if errors:
                LOGGER.warning("=" * 60)
                LOGGER.warning("ERRORS DETECTED. ABORTING SAVE TO PREVENT INCONSISTENCY.")
                for e in errors:
                    LOGGER.warning(f"  • {e}")
                LOGGER.warning("=" * 60)
                await smart_logout(page)
                return
 
            # --- SAVE: Click the Save button directly ---
            LOGGER.info("CLICKING SAVE BUTTON...")
            save_clicked = False
            for frame in page.frames:
                try:
                    # Target the explicit Save button in the WebGUI toolbar
                    save_btn = frame.locator("text=/^Save$/i, [title^='Save'], #btn\\[11\\]").first
                    if await save_btn.is_visible(timeout=1000):
                        await save_btn.click(force=True)
                        save_clicked = True
                        LOGGER.info("SAVE BUTTON CLICKED.")
                        break
                except: continue
 
            if not save_clicked:
                await page.keyboard.press("Control+s")
                LOGGER.info("SAVE via Ctrl+S.")
 
            # ── Situation-Based Dialog Handling ──
            # Specifically handle the "Prompt for customizing request" dialog and other post-save popups
            await asyncio.sleep(1.5)
            LOGGER.info("Waiting for post-save dialogs (e.g. Customizing Request)...")
            for attempt in range(5):
                found = await handle_sap_confirmation_dialogs(page, timeout_ms=2000)
                if found:
                    LOGGER.info(f"  ✔ Post-save dialog dismissed (Attempt {attempt+1}).")
                    await asyncio.sleep(2)
                    # Check again in case there's another dialog (like "Data saved" info)
                    continue
                else:
                    # Try a more specific selector for the green tick if handle_sap_confirmation_dialogs missed it
                    dialog_dismissed = False
                    for frame in page.frames:
                        try:
                            # Search for the button with the green tick icon or specific SAP WebGUI IDs
                            extra_selectors = [
                                "button[title*='Continue']", "button[title*='OK']",
                                "button[title*='Checkmark']", "#btn\\[0\\]",
                                ".sapMBtn:has-text('OK')", "button:has-text('Continue')"
                            ]
                            for sel in extra_selectors:
                                btn = frame.locator(sel).first
                                if await btn.is_visible(timeout=500):
                                    await btn.click(force=True)
                                    LOGGER.info(f"  ✔ Dialog dismissed via extra selector '{sel}'.")
                                    dialog_dismissed = True
                                    await asyncio.sleep(2)
                                    break
                            if dialog_dismissed: break
                        except: pass
                   
                    if not dialog_dismissed:
                        # If no dialog found after a few attempts, assume we're done
                        if attempt > 1:
                            break
                await asyncio.sleep(1)
 
            # ── Final Status Verification ──
            # Improved check: look specifically for success/error messages in the bottom status area
            await asyncio.sleep(1)
            webgui_frame = await get_webgui_frame(page) or webgui_frame
            msg_text = await webgui_frame.evaluate("""() => {
                const bar = document.querySelector('.lsStatusbar__message, [id*="msgarea"], .urMsgBarSucc');
                if (bar) return bar.innerText.trim();
                return "";
            }""")
 
            if msg_text:
                if "saved" in msg_text.lower():
                    LOGGER.info(f"✅ SUCCESS: {msg_text}")
                else:
                    LOGGER.info(f"📋 SAP Message: {msg_text}")
            else:
                LOGGER.info("No status bar message detected. Verifying if Save button is disabled...")
                # If save button is gone or disabled, it likely worked
                is_still_there = await webgui_frame.locator("text=/^Save$/i").first.is_visible(timeout=500)
                if not is_still_there:
                    LOGGER.info("✅ SUCCESS: Save button no longer visible (Data committed).")
                else:
                    LOGGER.warning("⚠️ Save button still visible. Save might have been rejected.")
 
            await smart_logout(page)
 
        except Exception as e:
            LOGGER.error(f"Task 3 Error: {e}", exc_info=True)
        finally:
            if not page.is_closed(): await browser.close()


async def Maintain_Budget_Availability_Control_Profile_for_Projects_102413(
    targets: list[dict],
):
    """
    Maintain Budget Availability Control Profile for Projects (SSCUI 102413)
    ───────────────────────────────────────────────────────────────────────────
    Columns (Tab order from Profile cell):
        Profile | Availy Ctrl Type (dropdown) | Budget Availability Ctrl Prfl Name (text) |
        *Time Range (dropdown) | *Budget Currency Type (dropdown)

    Target format:
        {
            "Profile": "PSQ01",
            "Availy Ctrl Type": "Project System",
            "Budget Availability Ctrl Prfl Name": "Projct Bdgt Avail Prfl Q Stk",
            "Time Range": "Annual Budget",
            "Budget Currency Type": "Global Currency"
        }
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FINSVC_AVC_PROF_PS&CustomizingObject=FINSVC_AVC_PROF_PS&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ER9_11001077&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context()
        page = await new_page_in_front(context)

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(5)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            LOGGER.info(f"WEBGUI FRAME FOUND. Processing {len(targets)} target(s)...")

            for idx, target in enumerate(targets, start=1):
                profile_id = str(target.get("Profile", "")).strip()
                availy_ctrl_type = target.get("Availy Ctrl Type", "")
                prfl_name = target.get("Budget Availability Ctrl Prfl Name", "")
                time_range = target.get("Time Range", "")
                budget_currency = target.get("Budget Currency Type", "")

                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING Profile={profile_id}")

                # ── Step 1: Find the row ──
                row_info = None
                for scroll_attempt in range(10):
                    if scroll_attempt == 0:
                        try:
                            pos_btn = webgui_frame.locator("text=/Position/i").first
                            if await pos_btn.is_visible(timeout=2000):
                                await pos_btn.click(force=True)
                                await asyncio.sleep(1)
                                await page.keyboard.type(profile_id, delay=50)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(2)
                        except:
                            pass

                    row_info = await webgui_frame.evaluate(
                        JS_FIND_ROW_BY_COCD, profile_id
                    )
                    if row_info:
                        LOGGER.info(
                            f"  Found row at prefix={row_info['prefix']}, row={row_info['rowIdx']}"
                        )
                        break

                    LOGGER.info(
                        f"  Not visible yet, PageDown ({scroll_attempt + 1}/10)..."
                    )
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(1.5)

                if not row_info:
                    LOGGER.error(f"  Profile '{profile_id}' NOT FOUND. Skipping.")
                    continue

                prefix = row_info["prefix"]
                row_idx = row_info["rowIdx"]
                col_idx = int(row_info["colIdx"])

                # ── Step 2: Click the Profile cell ──
                profile_cell_id = f"{prefix}[{row_idx},{col_idx}]"
                LOGGER.info(f"  Clicking Profile cell: {profile_cell_id}")
                try:
                    cell = webgui_frame.locator(f"id={profile_cell_id}")
                    if await cell.count() > 0:
                        await cell.first.click(force=True)
                    else:
                        await webgui_frame.evaluate(
                            f"() => document.getElementById('{profile_cell_id}')?.click()"
                        )
                except:
                    await webgui_frame.evaluate(
                        f"() => document.getElementById('{profile_cell_id}')?.click()"
                    )
                await asyncio.sleep(0.5)

                # ── Step 3: Tab through columns ──
                # Profile → Tab → Availy Ctrl Type → Tab → Budget Availability Ctrl Prfl Name → Tab → Time Range → Tab → Budget Currency Type

                # --- Col 2: Availy Ctrl Type (DROPDOWN) ---
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                if availy_ctrl_type:
                    LOGGER.info(f"  Setting Availy Ctrl Type → '{availy_ctrl_type}'")
                    await page.keyboard.type(availy_ctrl_type, delay=30)
                    await asyncio.sleep(0.5)
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(0.5)
                    LOGGER.info(f"  ✔ Availy Ctrl Type set.")

                # --- Col 3: Budget Availability Ctrl Prfl Name (TEXT) ---
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                if prfl_name:
                    LOGGER.info(f"  Setting Prfl Name → '{prfl_name}'")
                    await page.keyboard.down("Control")
                    await page.keyboard.press("a")
                    await page.keyboard.up("Control")
                    await page.keyboard.type(prfl_name, delay=30)
                    await asyncio.sleep(0.3)
                    LOGGER.info(f"  ✔ Prfl Name set.")

                # --- Col 4: Time Range (DROPDOWN) ---
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                if time_range:
                    LOGGER.info(f"  Setting Time Range → '{time_range}'")
                    await page.keyboard.type(time_range, delay=30)
                    await asyncio.sleep(0.5)
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(0.5)
                    LOGGER.info(f"  ✔ Time Range set.")

                # --- Col 5: Budget Currency Type (DROPDOWN) ---
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                if budget_currency:
                    LOGGER.info(f"  Setting Budget Currency Type → '{budget_currency}'")
                    await page.keyboard.type(budget_currency, delay=30)
                    await asyncio.sleep(0.5)
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(0.5)
                    LOGGER.info(f"  ✔ Budget Currency Type set.")

                # Confirm row entry
                await page.keyboard.press("Enter")
                await asyncio.sleep(1)
                LOGGER.info(f"  ✅ Profile '{profile_id}' — ALL FIELDS DONE.")

            # ── Step 4: Save ──
            LOGGER.info("SAVING ALL CHANGES...")
            await execute_save_flow(page)

            # Handle TWO post-save dialogs (searches ALL frames for tick buttons):
            # Dialog 1: "Display Logs" → click green ✅ tick
            # Dialog 2: "Prompt for customizing request" → click green ✓ tick
            for dialog_num in range(1, 3):
                await asyncio.sleep(2)
                LOGGER.info(f"Checking for post-save dialog #{dialog_num}...")
                if not await handle_sap_confirmation_dialogs(page, timeout_ms=5000):
                    LOGGER.info(
                        f"Dialog #{dialog_num} not found via selectors, pressing Enter..."
                    )
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(2)

            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"FATAL ERROR in 102413: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 4: Edit Tax Information for Company Codes (SSCUI 105675)
# ═══════════════════════════════════════════════════════════════════════════


async def edit_tax_information_for_company_codes_105675(targets: list[dict]):
    """
    Edit Tax Information for Company Codes (SSCUI 105675 / 11002281)
    ─────────────────────────────────────────────────────────────────
    Columns: CoCd | Company Name | C/R | Crcy | Tx Crcy Transl | VAT Reg No | Tax base | Tax Rept | Discount base
    """
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=V_001_CLD_TX&CustomizingObject=V_001_CLD_TX&"
        "CustomizingObjectType=V&CustomizingProject=&"
        "CustomizingTransaction=S_ER9_11002281&Type=SSCUI"
    )

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        async def _edit_cell(wf, cell_id, value):
            try:
                await wf.evaluate(
                    f"() => document.getElementById('{cell_id}')?.scrollIntoView({{ behavior: 'instant', block: 'center' }})"
                )
                await asyncio.sleep(0.2)
                loc = wf.locator(f"id={cell_id}")
                if await loc.count() == 0:
                    return False
                await loc.first.click(force=True, timeout=2000)
                await asyncio.sleep(0.3)
                await page.keyboard.press("F2")
                await asyncio.sleep(0.3)
                await wf.evaluate(f"""() => {{
                    let el = document.getElementById('{cell_id}');
                    if (!el) return;
                    let inp = (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') ? el : el.querySelector('input, textarea');
                    if (!inp) inp = document.activeElement;
                    if (inp && (inp.tagName === 'INPUT' || inp.tagName === 'TEXTAREA')) {{
                        inp.focus(); inp.value = ''; inp.dispatchEvent(new Event('input', {{bubbles: true}}));
                    }}
                }}""")
                await asyncio.sleep(0.2)
                await page.keyboard.type(str(value), delay=50)
                await asyncio.sleep(0.5)
                await page.keyboard.press("Tab")
                await asyncio.sleep(
                    1.5
                )  # Crucial wait for SAP background validation avoiding truncation of next field
                return True
            except Exception as e:
                LOGGER.warning(f"  _edit_cell failed for {cell_id}: {e}")
                return False

        async def _set_dropdown_cell(wf, cell_id, target_text):
            try:
                await wf.evaluate(
                    f"() => document.getElementById('{cell_id}')?.scrollIntoView({{ behavior: 'instant', block: 'center' }})"
                )
                await asyncio.sleep(0.2)
                loc = wf.locator(f"id={cell_id}")
                if await loc.count() == 0:
                    return False

                # Focus input and open dropdown
                await loc.first.click(force=True)
                await asyncio.sleep(0.5)
                await page.keyboard.press("Alt+ArrowDown")
                await asyncio.sleep(1.5)  # Wait for combo list popup

                target_lower = str(target_text).lower().strip()
                result = await wf.evaluate(
                    f"""(targetLower) => {{
                    const options = document.querySelectorAll('.lsListbox__value, .lsListbox__item, tr.urLsbDropRow td, .urLsbTxt, [role="option"]');
                    let bestMatch = null;

                    for (const opt of options) {{
                        const text = (opt.textContent || opt.innerText || '').trim().toLowerCase();
                        if (!text) continue;

                        if (text === targetLower) {{
                            opt.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                            opt.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true }}));
                            opt.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true }}));
                            opt.click();
                            return 'clicked_via_js';
                        }}

                        if (text.startsWith(targetLower) || targetLower.startsWith(text)) {{
                            bestMatch = opt;
                        }}
                    }}

                    if (bestMatch) {{
                        bestMatch.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        bestMatch.dispatchEvent(new MouseEvent('mousedown', {{ bubbles: true }}));
                        bestMatch.dispatchEvent(new MouseEvent('mouseup', {{ bubbles: true }}));
                        bestMatch.click();
                        return 'clicked_via_js';
                    }}
                    return null;
                }}""",
                    target_lower,
                )

                if result == "clicked_via_js":
                    LOGGER.info(
                        f"  Dropdown value matched and clicked via JS for {target_text}"
                    )
                else:
                    LOGGER.warning(
                        f"  Could not find dropdown option '{target_text}'. Falling back to typing."
                    )
                    await page.keyboard.press("Escape")
                    await asyncio.sleep(0.5)
                    await page.keyboard.type(str(target_text), delay=50)

                await asyncio.sleep(0.5)
                await page.keyboard.press("Tab")
                await asyncio.sleep(1.5)
                return True
            except Exception as e:
                LOGGER.warning(f"  _set_dropdown_cell failed for {cell_id}: {e}")
                return False

        async def _set_checkbox_cell(wf, cell_id, desired_state: bool):
            try:
                info = await wf.evaluate(f"""() => {{
                    const cellId = '{cell_id}';
                    let el = document.getElementById(cellId + '_c') || document.getElementById(cellId);
                    if (!el) return {{ status: 'not_found' }};

                    el.scrollIntoView({{ behavior: 'instant', block: 'center', inline: 'center' }});

                    let isChecked = false;
                    let targetClickId = el.id;

                    let chk = el.querySelector('input[type="checkbox"]');
                    if (!chk && el.tagName === 'INPUT' && el.type === 'checkbox') chk = el;

                    if (chk) {{
                        isChecked = chk.checked;
                        targetClickId = chk.id || el.id;
                    }} else {{
                        if (el.getAttribute('aria-checked') === 'true' || el.querySelector('[aria-checked="true"]')) {{
                            isChecked = true;
                        }} else if (el.classList.contains('urChkBxOn') || el.querySelector('.urChkBxOn')) {{
                            isChecked = true;
                        }} else {{
                            const raw = el.getAttribute('lsdata');
                            if (raw) {{
                                try {{
                                    const d = JSON.parse(raw.replace(/'/g, '"'));
                                    if (d['1'] === true || d['1'] === 'X') isChecked = true;
                                }} catch(e) {{}}
                            }}
                        }}
                    }}
                    return {{ status: 'found', is_checked: isChecked, target_id: targetClickId }};
                }}""")

                if info.get("status") == "not_found":
                    LOGGER.warning(f"  Checkbox {cell_id} not found.")
                    return "not_found"

                is_checked = info.get("is_checked", False)
                desired = bool(desired_state)

                if is_checked != desired:
                    target_id = info.get("target_id", cell_id)
                    loc = wf.locator(f"id={target_id}")
                    if await loc.count() > 0:
                        await loc.first.click(force=True)
                    else:
                        await wf.evaluate(
                            f"() => document.getElementById('{target_id}')?.click()"
                        )
                    await asyncio.sleep(0.5)
                    return "toggled"
                return "skip"
            except Exception as e:
                LOGGER.warning(f"  _set_checkbox failed for {cell_id}: {e}")
                return "error"

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(2)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                return

            errors = []
            for idx, target in enumerate(targets, start=1):
                cocd = str(target.get("CoCd", "")).strip()
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING CoCd={cocd}")

                row_info = None
                for scroll_attempt in range(10):
                    if scroll_attempt == 0:
                        try:
                            pos_btn = webgui_frame.locator("text=/Position/i").first
                            if await pos_btn.is_visible(timeout=1000):
                                await pos_btn.click(force=True)
                                await asyncio.sleep(0.5)
                                await page.keyboard.type(cocd)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(1.5)
                        except:
                            pass
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)
                    if row_info:
                        break
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(1)

                if not row_info:
                    LOGGER.error(f"  ✘ CoCd '{cocd}' not found in table.")
                    errors.append(f"CoCd '{cocd}' not found")
                    continue

                prefix = row_info["prefix"]
                r_idx = row_info["rowIdx"]
                c_idx = int(row_info["colIdx"])

                tx_crcy = target.get("Tx_Crcy_Transl")
                if tx_crcy is not None:
                    LOGGER.info(f"  Setting Tx Crcy Transl → '{tx_crcy}'")
                    await _set_dropdown_cell(
                        webgui_frame, f"{prefix}[{r_idx},{c_idx + 4}]", tx_crcy
                    )

                vat_reg = target.get("VAT_Reg_No")
                if vat_reg is not None:
                    vat_reg = str(vat_reg)[:20]
                    LOGGER.info(f"  Setting VAT Reg No → '{vat_reg}'")
                    await _edit_cell(
                        webgui_frame, f"{prefix}[{r_idx},{c_idx + 5}]", vat_reg
                    )

                tax_base = target.get("Tax_Base_Net")
                if tax_base is not None:
                    LOGGER.info(f"  Setting Tax base is net value → {tax_base}")
                    await _set_checkbox_cell(
                        webgui_frame, f"{prefix}[{r_idx},{c_idx + 6}]", bool(tax_base)
                    )

                tax_rept = target.get("Tax_Reporting_Date")
                if tax_rept is not None:
                    LOGGER.info(f"  Setting Tax Reporting Date Active → {tax_rept}")
                    await _set_checkbox_cell(
                        webgui_frame, f"{prefix}[{r_idx},{c_idx + 7}]", bool(tax_rept)
                    )

                disc_base = target.get("Discount_Base_Net")
                if disc_base is not None:
                    LOGGER.info(f"  Setting Discount base is net value → {disc_base}")
                    await _set_checkbox_cell(
                        webgui_frame, f"{prefix}[{r_idx},{c_idx + 8}]", bool(disc_base)
                    )

                LOGGER.info(f"  ✔ CoCd '{cocd}' edit complete.")

            if errors:
                LOGGER.warning("ERRORS DETECTED:")
                for e in errors:
                    LOGGER.warning(f"  • {e}")

            # execute_save_flow and graceful_exit already imported at module level
            await execute_save_flow(page)
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"Error in Tax Info automation: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


async def Define_Parameter_Sets_103635(targets: list[dict]):
    """
    Define Parameter Sets (SSCUI 103635)
    ─────────────────────────────────────
    Workflow: Click 'New Entries' → fill detail form via sequential TABs → 'Next Entry' → Save
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FARVC_BSIMP_PSETS&CustomizingObject=VC_FAR_BSIMP_PSETS&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ER9_11001563&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context()
        page = await new_page_in_front(context)

        async def _type_field(value):
            """Select all existing text → delete → type new value. Works for both empty and pre-filled fields."""
            await page.keyboard.down("Control")
            await page.keyboard.press("a")
            await page.keyboard.up("Control")
            await asyncio.sleep(0.1)
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.1)
            await page.keyboard.type(str(value), delay=20)
            await asyncio.sleep(0.3)  # Brief settle pause

        async def _set_checkbox(field_name, target_state, wf):
            """Tab to checkbox, detect state via webgui_frame, toggle if needed."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.5)  # Wait for checkbox to receive focus

            if target_state is None:
                LOGGER.info(f"  Checkbox '{field_name}': skipped (no target)")
                return

            # Detect state using webgui_frame (NOT page — the active element is inside the iframe!)
            JS_CHECKBOX_DETECT = """
            () => {
                const el = document.activeElement;
                if (!el) return null;

                // Strategy 1: Check lsdata on the element or parent
                const checkLsdata = (n) => {
                    if (!n) return null;
                    const raw = n.getAttribute('lsdata');
                    if (raw) {
                        try {
                            const d = JSON.parse(raw.replace(/'/g, '"'));
                            if (d['1'] === true || d['1'] === '1' || d['1'] === 1) return true;
                            if (d['1'] === false || d['1'] === '0' || d['1'] === 0 || d['1'] === '') return false;
                        } catch(e) {}
                    }
                    return null;
                };

                // Check the element itself
                let s = checkLsdata(el);
                if (s !== null) return s;

                // Check parent (SAP wraps checkboxes in a container)
                s = checkLsdata(el.parentElement);
                if (s !== null) return s;

                // Strategy 2: native checkbox
                if (el.tagName === 'INPUT' && el.type === 'checkbox') return el.checked;
                const inp = el.querySelector('input[type="checkbox"]');
                if (inp) return inp.checked;

                // Strategy 3: aria-checked
                const aria = el.getAttribute('aria-checked');
                if (aria === 'true') return true;
                if (aria === 'false') return false;

                // Strategy 4: check class names for "checked" state
                const cls = (el.className || '') + ' ' + (el.parentElement?.className || '');
                if (cls.includes('SAPBChk-Chk')) return true;
                if (cls.includes('SAPBChk-Uchk') || cls.includes('SAPBChk')) return false;

                return null;
            }
            """

            try:
                state = await wf.evaluate(JS_CHECKBOX_DETECT)
                LOGGER.info(
                    f"  Checkbox '{field_name}': detected={state}, target={target_state}"
                )

                if state is not None:
                    if state != target_state:
                        LOGGER.info(f"    → Toggling '{field_name}' (Space)")
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.3)
                    else:
                        LOGGER.info(f"    → No change needed for '{field_name}'")
                else:
                    LOGGER.warning(
                        f"    ! Detection failed for '{field_name}' — using fallback toggle"
                    )
                    # Fallback: for NEW entries, checkboxes default to UNCHECKED
                    # So toggle only if target is True
                    if target_state:
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.3)
            except Exception as e:
                LOGGER.warning(
                    f"    ! Error detecting '{field_name}': {e} — fallback toggle"
                )
                if target_state:
                    await page.keyboard.press("Space")
                    await asyncio.sleep(0.3)

            await asyncio.sleep(0.3)

        async def _click_next_entry():
            """Reliably click 'Next Entry' button with retries."""
            for attempt in range(3):
                for frame in page.frames:
                    try:
                        nxt = frame.locator("text=/Next Entry/i").first
                        if await nxt.is_visible(timeout=3000):
                            await nxt.click(force=True)
                            LOGGER.info(
                                f"    ✔ 'Next Entry' clicked (attempt {attempt + 1})"
                            )
                            await asyncio.sleep(2)
                            return True
                    except:
                        continue
                LOGGER.warning(
                    f"    ! 'Next Entry' not found (attempt {attempt + 1}/3), retrying..."
                )
                await asyncio.sleep(2)
            LOGGER.error("    !! 'Next Entry' FAILED after 3 attempts!")
            return False

        async def _fill_fields_2_to_8(name, method, ass_val, summ, zero, bundle, items, wf):
            """Fill fields 2-8 (P Set Name → Number of Items). Reused for edit and create."""
            LOGGER.info(f"  [2/8] P Set Name → '{name}'")
            await _type_field(name)

            await page.keyboard.press("Tab")
            await asyncio.sleep(0.4)
            LOGGER.info(f"  [3/8] Posting Method → '{method}'")
            await _type_field(method)

            LOGGER.info(f"  [4/8] Ass.Val.Dte to Acct → {ass_val}")
            await _set_checkbox("Ass.Val.Dte", ass_val, wf)

            LOGGER.info(f"  [5/8] Summarization → {summ}")
            await _set_checkbox("Summarization", summ, wf)

            LOGGER.info(f"  [6/8] ZeroSales Permitted → {zero}")
            await _set_checkbox("ZeroSales", zero, wf)

            await page.keyboard.press("Tab")
            await asyncio.sleep(0.4)
            LOGGER.info(f"  [7/8] Bundle Type → '{bundle}'")
            await _type_field(bundle)

            await page.keyboard.press("Tab")
            await asyncio.sleep(0.4)
            LOGGER.info(f"  [8/8] Number of Items → '{items if items else '[Empty]'}'")
            if items:
                await _type_field(items)
            else:
                await asyncio.sleep(0.3)

        async def _check_save_errors():
            """Check SAP status bar for errors after save. Returns error message or None."""
            wf_save = await get_webgui_frame(page)
            frames = [wf_save] if wf_save else []
            frames.extend([f for f in page.frames if f != wf_save])
            JS_ERR = """() => {
                const sels1 = '[class*="urMsgBarErr"],[class*="lsMsgBarErr"],[class*="Error"],.sapMMsgStripError';
                for (const icon of document.querySelectorAll(sels1)) {
                    const bar = icon.closest('[class*="msgbar" i],[class*="MsgBar"],[role="status"]') || icon.parentElement;
                    if (bar) { const t = (bar.textContent||'').trim(); if (t) return t.substring(0,500); }
                }
                const sels2 = '[id*="msgarea"],.lsMessageBar,.lsStatusbar__message,[role="status"]';
                for (const bar of document.querySelectorAll(sels2)) {
                    const t = (bar.textContent||'').trim(); if (!t) continue;
                    const h = bar.innerHTML||'';
                    if (h.includes('Err')||h.includes('error')||h.includes('Invalid')||h.includes('invalid'))
                        return t.substring(0,500);
                }
                return null;
            }"""
            for frame in frames:
                try:
                    msg = await frame.evaluate(JS_ERR)
                    if msg:
                        return msg
                except Exception:
                    continue
            return None

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(3)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            # ── Phase 1: For each target, Position-search then Edit or queue for creation ──
            to_create = []

            for idx, target in enumerate(targets, start=1):
                param_id = target.get("Parameter Set ID", "")
                name = target.get("P Set Name", "")
                method = target.get("Posting Method", "")
                ass_val = target.get("Ass.Val.Dte to Acct", None)
                summ = target.get("Summarization", None)
                zero = target.get("ZeroSales Permitted", None)
                bundle = target.get("Bundle Type", "")
                items = target.get("Number of Items", "")

                LOGGER.info(f"[{idx}/{len(targets)}] ═══ CONFIGURING: {param_id} ═══")

                # ── Try Position button to find existing entry ──
                found_existing = False
                try:
                    pos_btn = webgui_frame.locator("text=/Position/i").first
                    if await pos_btn.is_visible(timeout=3000):
                        await pos_btn.click(force=True)
                        await asyncio.sleep(0.5)
                        await page.keyboard.type(param_id, delay=20)
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(1)
                        # Check if the row now exists in visible table
                        row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, param_id)
                        if row_info:
                            found_existing = True
                            LOGGER.info(f"  ✔ ID '{param_id}' found in list — will edit existing entry")
                        else:
                            LOGGER.info(f"  ✖ ID '{param_id}' not found — will create new entry")
                except Exception as e:
                    LOGGER.warning(f"  Position search failed: {e}")

                if found_existing:
                    # ── EDIT existing entry: select row → Details → edit fields 2-8 ──
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, param_id)
                    prefix = row_info["prefix"]
                    row_idx = row_info["rowIdx"]
                    col_idx = row_info["colIdx"]
                    cell_id = f"{prefix}[{row_idx},{col_idx}]"

                    # Click on the row to select it
                    loc = webgui_frame.locator(f"id={cell_id}")
                    if await loc.count() > 0:
                        await loc.first.click(force=True)
                        await asyncio.sleep(0.5)
                    LOGGER.info(f"  ✔ Row selected for '{param_id}'")

                    # Click "Details" button to open detail form
                    details_opened = False
                    for frame in page.frames:
                        try:
                            btn = frame.locator("text=/^Details$/i").first
                            if await btn.is_visible(timeout=3000):
                                await btn.click(force=True)
                                await asyncio.sleep(2)
                                details_opened = True
                                LOGGER.info("  ✔ 'Details' view opened")
                                break
                        except:
                            continue

                    if not details_opened:
                        # Fallback: double-click the row
                        try:
                            await loc.first.dblclick(force=True)
                            await asyncio.sleep(2)
                            details_opened = True
                            LOGGER.info("  ✔ Details opened via double-click")
                        except:
                            pass

                    if not details_opened:
                        LOGGER.error(f"  !! Could not open details for '{param_id}'. Skipping.")
                        continue

                    # Re-acquire webgui frame for detail view
                    wf_detail = await get_webgui_frame(page) or webgui_frame

                    # In detail/edit mode, ID is read-only. Click ID field then Tab to P Set Name.
                    id_clicked = False
                    for frame in page.frames:
                        try:
                            el = frame.locator(
                                "input[lsdata*='txtFARV_BSIMP_PSETS-ID'], [lsdata*='txtFARV_BSIMP_PSETS-ID']"
                            ).first
                            if await el.is_visible(timeout=3000):
                                await el.click(force=True)
                                await asyncio.sleep(0.3)
                                id_clicked = True
                                break
                        except:
                            continue

                    if id_clicked:
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.4)
                    else:
                        LOGGER.warning("  Could not find ID field anchor, attempting Tab navigation...")
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.4)

                    # Fill fields 2-8
                    await _fill_fields_2_to_8(name, method, ass_val, summ, zero, bundle, items, wf_detail)

                    # Screenshot
                    await page.screenshot(path=f"/tmp/param_set_{param_id}.png")
                    LOGGER.info(f"  📸 Screenshot saved: /tmp/param_set_{param_id}.png")
                    LOGGER.info(f"  ✅ Record '{param_id}' EDIT COMPLETE.")

                    # Go back to list view
                    await page.keyboard.press("F3")
                    await asyncio.sleep(1.5)
                    # Re-acquire webgui_frame for next iteration
                    webgui_frame = await get_webgui_frame(page) or webgui_frame

                else:
                    # Queue for creation
                    to_create.append(target)

            # ── Phase 2: Create new entries (if any not found) ──
            if to_create:
                LOGGER.info(f"Creating {len(to_create)} new entries...")

                new_entry_clicked = False
                for frame in page.frames:
                    try:
                        btn = frame.locator("text=/New Entries/i").first
                        if await btn.is_visible(timeout=5000):
                            await btn.click(force=True)
                            new_entry_clicked = True
                            LOGGER.info("✔ 'New Entries' clicked.")
                            await asyncio.sleep(3)
                            break
                    except:
                        continue

                if not new_entry_clicked:
                    LOGGER.error("Could not find 'New Entries' button!")
                else:
                    for c_idx, target in enumerate(to_create, start=1):
                        param_id = target.get("Parameter Set ID", "")
                        name = target.get("P Set Name", "")
                        method = target.get("Posting Method", "")
                        ass_val = target.get("Ass.Val.Dte to Acct", None)
                        summ = target.get("Summarization", None)
                        zero = target.get("ZeroSales Permitted", None)
                        bundle = target.get("Bundle Type", "")
                        items = target.get("Number of Items", "")

                        LOGGER.info(f"[{c_idx}/{len(to_create)}] ═══ CREATING NEW: {param_id} ═══")

                        # Click ID field
                        anchor = None
                        for frame in page.frames:
                            try:
                                el = frame.locator(
                                    "input[lsdata*='txtFARV_BSIMP_PSETS-ID'], [lsdata*='txtFARV_BSIMP_PSETS-ID']"
                                ).first
                                if await el.is_visible(timeout=5000):
                                    await el.click(force=True)
                                    await asyncio.sleep(0.5)
                                    anchor = el
                                    LOGGER.info("  ✔ Clicked ID field (anchor)")
                                    break
                            except:
                                continue

                        if not anchor:
                            LOGGER.error("  !! No ID field found! Skipping.")
                            continue

                        # Field 1: Parameter Set ID
                        LOGGER.info(f"  [1/8] Parameter Set ID → '{param_id}'")
                        await _type_field(param_id)

                        # Tab to P Set Name, then fill fields 2-8
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.4)

                        wf_create = await get_webgui_frame(page) or webgui_frame
                        await _fill_fields_2_to_8(name, method, ass_val, summ, zero, bundle, items, wf_create)

                        # Screenshot
                        await page.screenshot(path=f"/tmp/param_set_{param_id}.png")
                        LOGGER.info(f"  📸 Screenshot saved: /tmp/param_set_{param_id}.png")
                        LOGGER.info(f"  ✅ Record '{param_id}' CREATE COMPLETE.")

                        # Next Entry if not last
                        if c_idx < len(to_create):
                            success = await _click_next_entry()
                            if not success:
                                LOGGER.error("  !! Cannot continue — 'Next Entry' failed. Stopping.")
                                break
                        else:
                            LOGGER.info("  Final new record reached.")

            # ── Step 3: Save and check for errors ──
            LOGGER.info("SAVING ALL CHANGES...")
            await execute_save_flow(page)
            await asyncio.sleep(2)

            error_message = await _check_save_errors()
            if error_message:
                try:
                    await page.screenshot(path="/tmp/save_error_103635.png")
                except Exception:
                    pass
                LOGGER.error(f"❌ SAVE ERROR DETECTED: {error_message}")
                LOGGER.error(f"📸 Error screenshot: /tmp/save_error_103635.png")
                LOGGER.error("⚠️  DATA WAS NOT SAVED — Please check the SAP screen for details.")
                await smart_logout(page)
                raise RuntimeError(f"SAP Save Error: {error_message}")
            else:
                LOGGER.info("✅ No error detected after save.")

            # Handle post-save dialogs (customizing request popups)
            for dialog_num in range(1, 3):
                await asyncio.sleep(1)
                LOGGER.info(f"Checking for post-save dialog #{dialog_num}...")
                if not await handle_sap_confirmation_dialogs(page, timeout_ms=5000):
                    LOGGER.info(f"Dialog #{dialog_num} not found, pressing Enter...")
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(1)

            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"FATAL ERROR in 103635: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


async def Define_Dunning_Block_Reasons_102259(targets: list[dict]):
    """
    Define Dunning Block Reasons (SSCUI 102259)
    ─────────────────────────────────────────────
    Table: Block | Text
    For each target:
      - Find Block row using JS_FIND_ROW_BY_COCD
      - Click the Text cell (blockCol+1) → F2 → clear → type new text → Tab
      - If Block not found → create via 'New Entries'

    Target format:  {"Block": "A", "Text": "Disputed"}
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFMENUORFBOB18&CustomizingObject=V_T040S&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87009499&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context()
        page = await new_page_in_front(context)

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(5)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            to_create = []

            # ── EDIT existing rows using proven cell-ID pattern ──
            for idx, target in enumerate(targets, start=1):
                block = target.get("Block", "").strip()
                text = target.get("Text", "")
                LOGGER.info(f"[{idx}/{len(targets)}] Block='{block}' → Text='{text}'")

                # Find the row via JS (same as all other functions)
                row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, block)

                if not row_info:
                    LOGGER.info(f"  ✖ Block '{block}' not found in table → will create")
                    to_create.append(target)
                    continue

                prefix = row_info["prefix"]
                row_idx = row_info["rowIdx"]
                block_col = int(row_info["colIdx"])
                text_col = block_col + 1

                LOGGER.info(
                    f"  Found: prefix={prefix}, row={row_idx}, blockCol={block_col}, textCol={text_col}"
                )

                # Construct the Text cell ID
                # Try both patterns: with _c suffix and without
                text_cell_id = f"{prefix}[{row_idx},{text_col}]"
                text_cell_id_c = f"{text_cell_id}_c"

                # Scroll into view
                await webgui_frame.evaluate(f"""() => {{
                    const el = document.getElementById('{text_cell_id}') || document.getElementById('{text_cell_id_c}');
                    if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                }}""")
                await asyncio.sleep(0.3)

                # Click the Text cell
                clicked = False
                for cid in [text_cell_id_c, text_cell_id]:
                    loc = webgui_frame.locator(f"id={cid}")
                    if await loc.count() > 0:
                        try:
                            await loc.first.click(force=True, timeout=2000)
                            clicked = True
                            LOGGER.info(f"  Clicked Text cell: {cid}")
                            break
                        except:
                            await webgui_frame.evaluate(
                                f"() => document.getElementById('{cid}')?.click()"
                            )
                            clicked = True
                            break

                if not clicked:
                    LOGGER.warning(
                        f"  Could not click Text cell. Skipping Block '{block}'."
                    )
                    continue

                await asyncio.sleep(0.3)

                # F2 to enter edit mode
                await page.keyboard.press("F2")
                await asyncio.sleep(0.3)

                # Clear the field via JS
                for cid in [text_cell_id_c, text_cell_id]:
                    await webgui_frame.evaluate(f"""() => {{
                        let cell = document.getElementById('{cid}');
                        if (!cell) return;
                        let inp = (cell.tagName === 'INPUT' || cell.tagName === 'TEXTAREA')
                            ? cell : cell.querySelector('input, textarea');
                        if (inp) {{
                            inp.focus();
                            inp.value = '';
                            inp.dispatchEvent(new Event('input', {{ bubbles: true }}));
                        }}
                    }}""")
                await asyncio.sleep(0.2)

                # Type the new text
                await page.keyboard.type(str(text), delay=50)
                await asyncio.sleep(0.3)

                # Tab to commit
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)

                LOGGER.info(f"  ✅ Block '{block}' → Text='{text}' DONE.")

            # ── CREATE new blocks via 'New Entries' (only if any not found) ──
            if to_create:
                LOGGER.info(
                    f"Creating {len(to_create)} new entries: {[t['Block'] for t in to_create]}"
                )

                for frame in page.frames:
                    try:
                        btn = frame.locator("text=/New Entries/i").first
                        if await btn.is_visible(timeout=5000):
                            await btn.click(force=True)
                            LOGGER.info("  ✔ 'New Entries' clicked.")
                            await asyncio.sleep(5)
                            break
                    except:
                        continue

                # Click first empty input to anchor
                for frame in page.frames:
                    try:
                        cell = frame.locator("input:visible").first
                        if await cell.is_visible(timeout=3000):
                            await cell.click(force=True)
                            await asyncio.sleep(1.5)
                            break
                    except:
                        continue

                for cidx, target in enumerate(to_create, start=1):
                    block = target.get("Block", "")
                    text = target.get("Text", "")
                    LOGGER.info(
                        f"  [CREATE {cidx}/{len(to_create)}] Block='{block}', Text='{text}'"
                    )

                    # Type Block
                    await page.keyboard.down("Control")
                    await page.keyboard.press("a")
                    await page.keyboard.up("Control")
                    await asyncio.sleep(0.2)
                    await page.keyboard.press("Delete")
                    await asyncio.sleep(0.2)
                    await page.keyboard.type(str(block), delay=80)
                    await asyncio.sleep(0.5)

                    # Tab to Text
                    await page.keyboard.press("Tab")
                    await asyncio.sleep(1)

                    # Type Text
                    await page.keyboard.down("Control")
                    await page.keyboard.press("a")
                    await page.keyboard.up("Control")
                    await asyncio.sleep(0.2)
                    await page.keyboard.press("Delete")
                    await asyncio.sleep(0.2)
                    await page.keyboard.type(str(text), delay=80)
                    await asyncio.sleep(0.5)

                    LOGGER.info(f"  ✅ Block '{block}' created.")

                    # Tab to next row
                    if cidx < len(to_create):
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(1)

            # ── Screenshot ──
            await page.screenshot(path="/tmp/dunning_block_reasons.png")
            LOGGER.info("📸 Screenshot saved: /tmp/dunning_block_reasons.png")

            # ── Save ──
            LOGGER.info("SAVING...")
            await execute_save_flow(page)
            for d in range(1, 3):
                await asyncio.sleep(2)
                if not await handle_sap_confirmation_dialogs(page, timeout_ms=5000):
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(2)

            # ── Exit ──
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"FATAL ERROR in 102259: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 5: Set Up Paying Company Codes for Payment Transactions (SSCUI 101001)
# ═══════════════════════════════════════════════════════════════════════════

# JS: Find an input/textarea near a label by scanning for label text
JS_FIND_INPUT_BY_LABEL = """
(labelText) => {
    const labelLower = labelText.toLowerCase().trim();

    // Helper: check if an input is a valid form field (not toolbar/navigation)
    function isValidInput(inp) {
        if (!inp || !inp.id) return false;
        const id = inp.id.toLowerCase();
        // Exclude toolbar, search, navigation inputs
        if (id.includes('toolbar') || id.includes('okcode') || id.includes('search')) return false;
        // Exclude inputs in toolbar containers
        if (inp.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) return false;
        // Must be a text-like input
        const type = (inp.type || '').toLowerCase();
        if (type === 'checkbox' || type === 'radio' || type === 'hidden' || type === 'submit' || type === 'button') return false;
        return true;
    }

    // Strategy 1: Find <label> or <span> with matching text, then find nearby input
    const allLabels = document.querySelectorAll('label, span, td');
    for (const lbl of allLabels) {
        const txt = (lbl.textContent || '').trim().toLowerCase();
        if (!txt.includes(labelLower)) continue;
        // Avoid matching on section headers or large containers
        if (lbl.children.length > 3) continue;
        // Skip if this element is inside the toolbar
        if (lbl.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) continue;

        // Strategy 1a: Check for 'for' attribute on <label>
        if (lbl.tagName === 'LABEL' && lbl.htmlFor) {
            const target = document.getElementById(lbl.htmlFor);
            if (target && isValidInput(target)) return { id: target.id, tag: target.tagName, value: target.value || '' };
        }

        // Strategy 1b: Walk parent row for an input
        let row = lbl.closest('tr') || lbl.parentElement;
        if (row) {
            const inputs = row.querySelectorAll('input, textarea');
            for (const inp of inputs) {
                if (isValidInput(inp)) return { id: inp.id, tag: inp.tagName, value: inp.value || '' };
            }
        }

        // Strategy 1c: Check next sibling elements
        let sib = lbl.nextElementSibling;
        for (let i = 0; i < 5 && sib; i++) {
            if (sib.querySelector) {
                const inputs = sib.querySelectorAll('input, textarea');
                for (const inp of inputs) {
                    if (isValidInput(inp)) return { id: inp.id, tag: inp.tagName, value: inp.value || '' };
                }
            }
            if ((sib.tagName === 'INPUT' || sib.tagName === 'TEXTAREA') && isValidInput(sib)) {
                return { id: sib.id, tag: sib.tagName, value: sib.value || '' };
            }
            sib = sib.nextElementSibling;
        }
    }
    return null;
}
"""

# JS: Find a checkbox near a label and return its state
JS_FIND_CHECKBOX_BY_LABEL = """
(labelText) => {
    const labelLower = labelText.toLowerCase().trim();
    const allLabels = document.querySelectorAll('label, span, td, div');

    for (const lbl of allLabels) {
        const txt = (lbl.textContent || '').trim().toLowerCase();
        if (!txt.includes(labelLower)) continue;
        if (lbl.children.length > 3) continue;

        // Check if the label itself IS a checkbox wrapper
        let chk = lbl.querySelector('input[type="checkbox"]');
        if (chk) return { id: chk.id, checked: chk.checked, clickId: chk.id };

        // Check for SAP-style checkbox spans (urChkBx)
        let sapChk = lbl.querySelector('[class*="urChkBx"], [role="checkbox"]');
        if (sapChk) {
            const isChecked = sapChk.classList.contains('urChkBxOn') ||
                              sapChk.getAttribute('aria-checked') === 'true';
            return { id: sapChk.id, checked: isChecked, clickId: sapChk.id };
        }

        // Walk parent row
        let row = lbl.closest('tr') || lbl.parentElement;
        if (row) {
            chk = row.querySelector('input[type="checkbox"]');
            if (chk) return { id: chk.id, checked: chk.checked, clickId: chk.id };

            sapChk = row.querySelector('[class*="urChkBx"], [role="checkbox"]');
            if (sapChk) {
                const isChecked = sapChk.classList.contains('urChkBxOn') ||
                                  sapChk.getAttribute('aria-checked') === 'true';
                return { id: sapChk.id, checked: isChecked, clickId: sapChk.id };
            }
        }

        // Walk siblings
        let sib = lbl.nextElementSibling;
        for (let i = 0; i < 5 && sib; i++) {
            chk = sib.querySelector ? sib.querySelector('input[type="checkbox"]') : null;
            if (chk) return { id: chk.id, checked: chk.checked, clickId: chk.id };
            if (sib.tagName === 'INPUT' && sib.type === 'checkbox') return { id: sib.id, checked: sib.checked, clickId: sib.id };

            sapChk = sib.querySelector ? sib.querySelector('[class*="urChkBx"], [role="checkbox"]') : null;
            if (sapChk) {
                const isChecked = sapChk.classList.contains('urChkBxOn') ||
                                  sapChk.getAttribute('aria-checked') === 'true';
                return { id: sapChk.id, checked: isChecked, clickId: sapChk.id };
            }
            sib = sib.nextElementSibling;
        }

        // Check preceding sibling (some SAP layouts put checkbox before label)
        let prev = lbl.previousElementSibling;
        for (let i = 0; i < 3 && prev; i++) {
            chk = prev.querySelector ? prev.querySelector('input[type="checkbox"]') : null;
            if (chk) return { id: chk.id, checked: chk.checked, clickId: chk.id };
            if (prev.tagName === 'INPUT' && prev.type === 'checkbox') return { id: prev.id, checked: prev.checked, clickId: prev.id };

            sapChk = prev.querySelector ? prev.querySelector('[class*="urChkBx"], [role="checkbox"]') : null;
            if (sapChk) {
                const isChecked = sapChk.classList.contains('urChkBxOn') ||
                                  sapChk.getAttribute('aria-checked') === 'true';
                return { id: sapChk.id, checked: isChecked, clickId: sapChk.id };
            }
            prev = prev.previousElementSibling;
        }
    }
    return null;
}
"""

# JS: Find a radio button by its label text
JS_FIND_RADIO_BY_LABEL = """
(labelText) => {
    const labelLower = labelText.toLowerCase().trim();
    const allLabels = document.querySelectorAll('label, span, td, div');

    for (const lbl of allLabels) {
        const txt = (lbl.textContent || '').trim().toLowerCase();
        if (!txt.includes(labelLower)) continue;
        if (lbl.children.length > 3) continue;

        // Check for <input type="radio"> inside or nearby
        let radio = lbl.querySelector('input[type="radio"]');
        if (radio) return { id: radio.id, checked: radio.checked };

        // SAP-style radio spans
        let sapRadio = lbl.querySelector('[class*="urRdb"], [role="radio"]');
        if (sapRadio) {
            const isChecked = sapRadio.classList.contains('urRdbOn') ||
                              sapRadio.getAttribute('aria-checked') === 'true';
            return { id: sapRadio.id, checked: isChecked };
        }

        // Walk parent row
        let row = lbl.closest('tr') || lbl.parentElement;
        if (row) {
            radio = row.querySelector('input[type="radio"]');
            if (radio) return { id: radio.id, checked: radio.checked };

            sapRadio = row.querySelector('[class*="urRdb"], [role="radio"]');
            if (sapRadio) {
                const isChecked = sapRadio.classList.contains('urRdbOn') ||
                                  sapRadio.getAttribute('aria-checked') === 'true';
                return { id: sapRadio.id, checked: isChecked };
            }
        }

        // Walk siblings
        let sib = lbl.nextElementSibling;
        for (let i = 0; i < 5 && sib; i++) {
            radio = sib.querySelector ? sib.querySelector('input[type="radio"]') : null;
            if (radio) return { id: radio.id, checked: radio.checked };
            if (sib.tagName === 'INPUT' && sib.type === 'radio') return { id: sib.id, checked: sib.checked };
            sib = sib.nextElementSibling;
        }

        // Check preceding sibling
        let prev = lbl.previousElementSibling;
        for (let i = 0; i < 3 && prev; i++) {
            radio = prev.querySelector ? prev.querySelector('input[type="radio"]') : null;
            if (radio) return { id: radio.id, checked: radio.checked };
            if (prev.tagName === 'INPUT' && prev.type === 'radio') return { id: prev.id, checked: prev.checked };
            prev = prev.previousElementSibling;
        }
    }
    return null;
}
"""

# JS: Find a text input by label within a specific section (for disambiguating duplicate labels)
JS_FIND_INPUT_BY_LABEL_IN_SECTION = """
(args) => {
    const labelLower = args.label.toLowerCase().trim();
    const sectionLower = args.section.toLowerCase().trim();

    function isValid(inp) {
        if (!inp || !inp.id) return false;
        const id = inp.id.toLowerCase();
        if (id.includes('toolbar') || id.includes('okcode') || id.includes('search') || id.includes('quickconf')) return false;
        if (inp.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) return false;
        const rect = inp.getBoundingClientRect();
        if (rect.width <= 5 || rect.height <= 5) return false;
        return true;
    }

    // STEP 1: Find the section heading - prefer LEAF-level elements with shortest match
    const headings = document.querySelectorAll('span, div, h1, h2, h3, h4, h5, h6, td, fieldset, legend, [role="heading"]');
    let sectionEl = null;
    let bestSectionLen = 99999;

    for (const h of headings) {
        const txt = (h.textContent || '').trim().toLowerCase();
        if (txt === sectionLower || txt.includes(sectionLower)) {
            if (h.children.length > 5) continue;
            // Prefer the shortest text match (most specific / leaf-level heading)
            if (txt.length < bestSectionLen) {
                bestSectionLen = txt.length;
                sectionEl = h;
            }
        }
    }
    if (!sectionEl) return null;

    // Scroll section heading into view so Y-coordinates are valid
    sectionEl.scrollIntoView({ behavior: 'instant', block: 'start' });

    // STEP 2: Find the matching label BELOW the section heading using Y-coordinates
    const sectionRect = sectionEl.getBoundingClientRect();
    const sectionBottomY = sectionRect.bottom;

    const allLabels = document.querySelectorAll('span, label, td, div.urLbl');
    let targetLabelEl = null;
    let bestLabelDist = 99999;

    for (const el of allLabels) {
        if (el.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) continue;

        // Get direct text (leaf-level matching)
        let directText = '';
        for (const cn of el.childNodes) {
            if (cn.nodeType === 3) directText += cn.textContent;
        }
        directText = directText.trim().toLowerCase();
        const fullText = (el.innerText || el.textContent || '').trim().toLowerCase();

        let matched = false;
        if (directText.includes(labelLower) && directText.length < labelLower.length + 20) {
            matched = true;
        } else if (el.children.length === 0 && fullText.includes(labelLower) && fullText.length < labelLower.length + 15) {
            matched = true;
        } else if (el.children.length <= 1 && fullText.includes(labelLower) && fullText.length < labelLower.length + 10) {
            matched = true;
        }

        if (!matched) continue;

        const elRect = el.getBoundingClientRect();
        // Label must be BELOW the section heading
        if (elRect.top < sectionBottomY - 5) continue;
        // Must be visible
        if (elRect.width <= 0 || elRect.height <= 0) continue;

        // Pick the closest label below the section heading
        const dist = elRect.top - sectionBottomY;
        if (dist < bestLabelDist) {
            bestLabelDist = dist;
            targetLabelEl = el;
        }
    }

    if (!targetLabelEl) return null;

    // STEP 3: Find input on the same Y-row as the label
    const labelRect = targetLabelEl.getBoundingClientRect();
    const labelCenterY = labelRect.top + labelRect.height / 2;

    const allInputs = Array.from(document.querySelectorAll('input:not([type="hidden"]), textarea'));
    let bestInp = null;
    let minDistance = 9999;

    for (const inp of allInputs) {
        if (!isValid(inp)) continue;

        const inpRect = inp.getBoundingClientRect();
        const inpCenterY = inpRect.top + inpRect.height / 2;

        const distY = Math.abs(labelCenterY - inpCenterY);
        const distX = inpRect.left - labelRect.right;

        if (distY < 15 && distX > -50 && distX < 600) {
            if (distY < minDistance) {
                minDistance = distY;
                bestInp = inp;
            }
        }
    }

    if (bestInp) {
        return { id: bestInp.id, tag: bestInp.tagName, y: labelRect.top };
    }

    // Fallback: row-based
    let row = targetLabelEl.closest('tr') || targetLabelEl.parentElement;
    if (row) {
        const fallbackInp = row.querySelector('input:not([type="hidden"]), textarea');
        if (fallbackInp && isValid(fallbackInp)) return { id: fallbackInp.id, tag: fallbackInp.tagName };
    }

    return null;
}
"""

# JS: Check if a field has an error state (red border or SAP error class)
JS_CHECK_FIELD_ERROR = """
(elementId) => {
    const el = document.getElementById(elementId);
    if (!el) return false;
    
    // Check classes
    const cls = el.className || '';
    if (cls.includes('urErr') || cls.includes('sapMInputError')) return true;
    
    // Check parent/container classes
    let p = el.parentElement;
    for (let i = 0; i < 3 && p; i++) {
        if (p.className && (p.className.includes('urErr') || p.className.includes('sapMInputError'))) return true;
        p = p.parentElement;
    }
    
    // Check computed style for red border
    const style = window.getComputedStyle(el);
    if (style.borderColor === 'rgb(255, 0, 0)' || style.borderColor.includes('red')) return true;
    
    return false;
}
"""


async def setup_paying_company_codes_101001(targets: list[dict]):
    """
    Set Up Paying Company Codes for Payment Transactions (SSCUI 101001)
    ────────────────────────────────────────────────────────────────────
    List view: Paying Company Code | Name
    Detail view: Control Data, SEPA Payments, Bill of Exchange Data (optional)

    Target format:
        {
            "CoCd": "1810",
            "Min_Incoming_Payment": "0.00",
            "Min_Outgoing_Payment": "0.00",
            "No_Exchange_Rate_Diff": True,
            "No_Exch_Rate_Diff_Part": False,
            "Separate_Payment_Each_Ref": False,
            "Bill_Exch_Pymt": False,
            "Direct_Debit_Pre_Notifications": False,
            "Creditor_ID_Number": "",
            "Create_Bills_Exchange": "per_due_date",
            "Latest_Due_Date_Incoming": "",
            "Bill_On_Demand_Due_Date": "",
            "Earliest_Due_Date_Outgoing": "1",
            "Latest_Due_Date_Outgoing": "999",
        }
    All fields except CoCd are optional — omit or set to None to skip.
    """
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=FIAPPY_V_T042B&CustomizingObject=V_T042B&"
        "CustomizingObjectType=V&CustomizingProject=&"
        "CustomizingTransaction=S_ALR_87100688&Type=SSCUI"
    )
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        # ── Local helper: type value into an input found by label ──
        async def _set_field_by_label(wf, label_text, value):
            """Find an input by its label text, clear it, and type the new value, with verification."""
            try:
                info = await wf.evaluate(JS_FIND_INPUT_BY_LABEL, label_text)
                if not info:
                    LOGGER.warning(
                        f"    Input for '{label_text}' not found on this screen. Skipping."
                    )
                    return False
                input_id = info["id"]
                LOGGER.info(f"    Found input for '{label_text}' → #{input_id}")

                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{input_id}');
                    if (el) {{
                        el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        el.focus();
                    }}
                }}""")
                await asyncio.sleep(0.3)
                
                loc = wf.locator(f"id={input_id}")
                await loc.first.click(force=True)
                await asyncio.sleep(0.2)

                await page.keyboard.down("Control")
                await page.keyboard.press("a")
                await page.keyboard.up("Control")
                await page.keyboard.press("Delete")
                await page.keyboard.type(str(value), delay=30)
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.8)

                # --- VERIFICATION ---
                actual_value = await loc.first.input_value()
                if str(value).strip() in actual_value.strip() or actual_value.strip() in str(value).strip():
                    LOGGER.info(f"    ✔ Set '{label_text}' = '{actual_value}' (Verified)")
                else:
                    LOGGER.warning(f"    ⚠ Verification FAILED for '{label_text}'. Expected '{value}', got '{actual_value}'")

                # --- ERROR DETECTION ---
                has_error = await wf.evaluate(JS_CHECK_FIELD_ERROR, input_id)
                if has_error:
                    LOGGER.error(f"    ❌ Validation Error detected on field '{label_text}'!")
                    status = await get_status_bar_message(wf)
                    if status:
                        LOGGER.error(f"      Status Message: {status.get('text')}")

                return True
            except Exception as e:
                LOGGER.warning(f"    Failed to set '{label_text}': {e}")
                return False

        # ── Local helper: type value into an input found by label in a specific section ──
        async def _set_field_by_label_in_section(wf, label_text, section_text, value):
            """Find an input by label within a specific section heading context, with verification."""
            try:
                info = await wf.evaluate(
                    JS_FIND_INPUT_BY_LABEL_IN_SECTION,
                    {"label": label_text, "section": section_text},
                )
                if not info:
                    LOGGER.warning(
                        f"    Input for '{label_text}' in section '{section_text}' not found. Skipping."
                    )
                    return False
                input_id = info["id"]
                LOGGER.info(
                    f"    Found input for '{label_text}' in '{section_text}' → #{input_id}"
                )
                
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{input_id}');
                    if (el) {{
                        el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        el.focus();
                    }}
                }}""")
                await asyncio.sleep(0.3)

                loc = wf.locator(f"id={input_id}")
                await loc.first.click(force=True)
                await asyncio.sleep(0.2)

                # Clear and type
                await page.keyboard.down("Control")
                await page.keyboard.press("a")
                await page.keyboard.up("Control")
                await page.keyboard.press("Delete")
                await page.keyboard.type(str(value), delay=30)
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.8) # Wait for SAP to process entry

                # --- VERIFICATION ---
                actual_value = await loc.first.input_value()
                # Normalize values (e.g. "500" vs "500.00")
                if str(value).strip() in actual_value.strip() or actual_value.strip() in str(value).strip():
                    LOGGER.info(f"    ✔ Set '{label_text}' in '{section_text}' = '{actual_value}' (Verified)")
                else:
                    LOGGER.warning(f"    ⚠ Verification FAILED for '{label_text}'. Expected '{value}', got '{actual_value}'")

                # --- ERROR DETECTION ---
                has_error = await wf.evaluate(JS_CHECK_FIELD_ERROR, input_id)
                if has_error:
                    LOGGER.error(f"    ❌ Validation Error detected on field '{label_text}'!")
                    # Check status bar for more info
                    status = await get_status_bar_message(wf)
                    if status:
                        LOGGER.error(f"      Status Message: {status.get('text')}")
                
                return True
            except Exception as e:
                LOGGER.warning(
                    f"    Failed to set '{label_text}' in section '{section_text}': {e}"
                )
                return False

        # ── Local helper: toggle a checkbox found by label ──
        async def _set_checkbox_by_label(wf, label_text, desired_state: bool):
            """Find a checkbox by its label text and set it to desired state, with verification."""
            try:
                info = await wf.evaluate(JS_FIND_CHECKBOX_BY_LABEL, label_text)
                if not info:
                    LOGGER.warning(
                        f"    Checkbox for '{label_text}' not found on this screen. Skipping."
                    )
                    return False

                click_id = info["clickId"]
                is_checked = info.get("checked", False)

                LOGGER.info(
                    f"    Checkbox '{label_text}': found #{click_id}, current={is_checked}, desired={desired_state}"
                )

                if is_checked == desired_state:
                    LOGGER.info(
                        f"    ✔ '{label_text}' already {'checked' if is_checked else 'unchecked'}. Skipping."
                    )
                    return True

                LOGGER.info(
                    f"    Toggling '{label_text}' from {is_checked} → {desired_state}"
                )

                # Scroll into view
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{click_id}');
                    if (el) el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                }}""")
                await asyncio.sleep(0.3)

                # Attempt 1: Click the element directly
                loc = wf.locator(f"id={click_id}")
                if await loc.count() > 0:
                    await loc.first.click(force=True)
                else:
                    await wf.evaluate(
                        f"() => document.getElementById('{click_id}')?.click()"
                    )
                await asyncio.sleep(0.8)

                # --- POST-CLICK VERIFICATION ---
                info_after = await wf.evaluate(JS_FIND_CHECKBOX_BY_LABEL, label_text)
                new_state = info_after.get("checked", False) if info_after else is_checked

                if new_state == desired_state:
                    LOGGER.info(f"    ✔ Toggled '{label_text}' → {desired_state} (Verified)")
                    return True

                LOGGER.warning(f"    ⚠ Click did not toggle '{label_text}'. Retrying with parent click + Space...")

                # Attempt 2: Click the parent element (SAP wraps checkboxes in <span>)
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{click_id}');
                    if (el && el.parentElement) {{
                        el.parentElement.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        el.parentElement.click();
                    }}
                }}""")
                await asyncio.sleep(0.5)

                info_after2 = await wf.evaluate(JS_FIND_CHECKBOX_BY_LABEL, label_text)
                new_state2 = info_after2.get("checked", False) if info_after2 else is_checked

                if new_state2 == desired_state:
                    LOGGER.info(f"    ✔ Toggled '{label_text}' → {desired_state} (Verified via parent click)")
                    return True

                # Attempt 3: Focus + Space key
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{click_id}');
                    if (el) {{ el.focus(); }}
                }}""")
                await asyncio.sleep(0.2)
                await page.keyboard.press("Space")
                await asyncio.sleep(0.8)

                info_after3 = await wf.evaluate(JS_FIND_CHECKBOX_BY_LABEL, label_text)
                new_state3 = info_after3.get("checked", False) if info_after3 else is_checked

                if new_state3 == desired_state:
                    LOGGER.info(f"    ✔ Toggled '{label_text}' → {desired_state} (Verified via Space key)")
                    return True

                LOGGER.error(f"    ❌ FAILED to toggle '{label_text}' after 3 attempts. State remains {new_state3}.")
                return False
            except Exception as e:
                LOGGER.warning(f"    Failed to toggle '{label_text}': {e}")
                return False

        # ── Local helper: click a radio button found by label ──
        async def _set_radio_by_label(wf, label_text):
            """Find a radio button by its label text and select it, with verification."""
            try:
                info = await wf.evaluate(JS_FIND_RADIO_BY_LABEL, label_text)
                if not info:
                    LOGGER.warning(
                        f"    Radio for '{label_text}' not found on this screen. Skipping."
                    )
                    return False

                radio_id = info["id"]
                is_checked = info.get("checked", False)

                LOGGER.info(
                    f"    Radio '{label_text}': found #{radio_id}, current={is_checked}"
                )

                if is_checked:
                    LOGGER.info(
                        f"    ✔ Radio '{label_text}' already selected. Skipping."
                    )
                    return True

                LOGGER.info(f"    Selecting radio '{label_text}' → #{radio_id}")

                # Use the proven Playwright native-click approach
                found = await wf.evaluate(f"""() => {{
                    document.querySelectorAll('[data-pw-radio-target]').forEach(
                        e => e.removeAttribute('data-pw-radio-target')
                    );
                    const radio = document.getElementById('{radio_id}');
                    if (!radio) return false;

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

                    clickTarget.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                    clickTarget.setAttribute('data-pw-radio-target', 'true');
                    return true;
                }}""")

                if not found:
                    LOGGER.warning(f"    Radio #{radio_id} not found in DOM.")
                    return False

                target = wf.locator('[data-pw-radio-target="true"]').first
                if await target.count() > 0:
                    await target.click(force=True)
                    await asyncio.sleep(1)

                    # --- POST-CLICK VERIFICATION ---
                    info_after = await wf.evaluate(JS_FIND_RADIO_BY_LABEL, label_text)
                    if info_after and info_after.get("checked", False):
                        LOGGER.info(f"    ✔ Radio '{label_text}' selected (Verified)")
                        return True

                    LOGGER.warning(f"    ⚠ Click did not select radio '{label_text}'. Retrying with direct click...")

                # Fallback: force-click the radio itself
                radio_loc = wf.locator(f"id={radio_id}")
                if await radio_loc.count() > 0:
                    await radio_loc.first.click(force=True)
                    await asyncio.sleep(1)

                    info_after2 = await wf.evaluate(JS_FIND_RADIO_BY_LABEL, label_text)
                    if info_after2 and info_after2.get("checked", False):
                        LOGGER.info(f"    ✔ Radio '{label_text}' selected (Verified via direct click)")
                        return True

                # Fallback: Space key
                await wf.evaluate(f"""() => {{
                    const el = document.getElementById('{radio_id}');
                    if (el) el.focus();
                }}""")
                await asyncio.sleep(0.2)
                await page.keyboard.press("Space")
                await asyncio.sleep(0.8)

                info_after3 = await wf.evaluate(JS_FIND_RADIO_BY_LABEL, label_text)
                if info_after3 and info_after3.get("checked", False):
                    LOGGER.info(f"    ✔ Radio '{label_text}' selected (Verified via Space key)")
                    return True

                LOGGER.error(f"    ❌ FAILED to select radio '{label_text}' after 3 attempts.")
                return False
            except Exception as e:
                LOGGER.warning(f"    Failed to select radio '{label_text}': {e}")
                return False

        # ════════════════════════════════════════════════════════════════
        # MAIN FLOW
        # ════════════════════════════════════════════════════════════════
        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(3)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            for idx, target in enumerate(targets, start=1):
                if page.is_closed():
                    LOGGER.error("Browser closed. Stopping.")
                    break

                cocd = str(target.get("CoCd", "")).strip()
                LOGGER.info(f"[{idx}/{len(targets)}] PROCESSING CoCd={cocd}")

                # Re-acquire frame (may change after navigation)
                webgui_frame = await get_webgui_frame(page) or page

                # ── Step 1: Find row in list view ──
                row_info = None
                for scroll_attempt in range(10):
                    if scroll_attempt == 0:
                        try:
                            pos_btn = webgui_frame.locator("text=/Position/i").first
                            if await pos_btn.is_visible(timeout=1500):
                                await pos_btn.click(force=True)
                                await asyncio.sleep(0.5)
                                await page.keyboard.type(cocd, delay=50)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(1.5)
                        except:
                            pass
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)
                    if row_info:
                        break
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(1)

                if not row_info:
                    LOGGER.error(f"  ✘ CoCd '{cocd}' not found in list. Skipping.")
                    continue

                prefix = row_info["prefix"]
                r_idx = row_info["rowIdx"]
                c_idx = int(row_info["colIdx"])

                LOGGER.info(f"  Found row: prefix={prefix}, row={r_idx}, col={c_idx}")

                # ── Step 2: Click the Name cell to select the row ──
                name_cell_id = f"{prefix}[{r_idx},{c_idx + 1}]"
                LOGGER.info(f"  Clicking Name cell: {name_cell_id}")
                try:
                    cell = webgui_frame.locator(f"id={name_cell_id}")
                    if await cell.count() > 0:
                        await cell.first.click(force=True)
                    else:
                        await webgui_frame.evaluate(
                            f"() => document.getElementById('{name_cell_id}')?.click()"
                        )
                except:
                    await webgui_frame.evaluate(
                        f"() => document.getElementById('{name_cell_id}')?.click()"
                    )
                await asyncio.sleep(0.5)

                # ── Step 3: Click Details button ──
                LOGGER.info("  Clicking 'Details' button...")
                details_clicked = False
                for det_text in ["Details", "Detail"]:
                    try:
                        det_btn = webgui_frame.locator(f"text='{det_text}'").first
                        if await det_btn.is_visible(timeout=1500):
                            await det_btn.click(force=True)
                            details_clicked = True
                            LOGGER.info(f"  Clicked '{det_text}' button.")
                            break
                    except:
                        continue

                if not details_clicked:
                    # Fallback: try Ctrl+Shift+F2 or F2 for detail navigation
                    LOGGER.info("  Details button not found. Trying Ctrl+Shift+F2...")
                    await page.keyboard.press("Control+Shift+F2")

                await asyncio.sleep(3)

                # Verify we're on the details screen
                webgui_frame = await get_webgui_frame(page) or page
                try:
                    still_list = await webgui_frame.locator(
                        "text=/Position/i"
                    ).first.is_visible(timeout=2000)
                except:
                    still_list = False

                if still_list:
                    LOGGER.error(
                        f"  ✘ Could not enter details for CoCd '{cocd}'. Skipping."
                    )
                    continue

                LOGGER.info("  ✔ Entered detail view.")

                # ── Entry Logic ──
                control_data = target.get("Control Data", target)
                sepa_data = target.get("Specifications for SEPA Payments", target)
                boe_data = target.get("Bill of Exchange Data", target)
                # Radio buttons
                create_boe_section = boe_data.get("Create Bills of Exchange", {})
                if isinstance(create_boe_section, dict):
                    boe_radios = [
                        ("One Bill of Exchange per Invoice", "One Bill of Exchange per Invoice"),
                        ("One Bill of Exchange per Due Date", "One Bill of Exchange per Due Date"),
                        ("One Bill of Exch. per Due Date Per.", "One Bill of Exch. per Due Date Per"),
                    ]
                    for json_key, label in boe_radios:
                        if create_boe_section.get(json_key):
                            await _set_radio_by_label(webgui_frame, label)
                            break
                else:
                    # Legacy flat radio handling
                    create_boe = target.get("Create_Bills_Exchange")
                    if create_boe:
                        radio_map = {
                            "per_invoice": "One Bill of Exchange per Invoice",
                            "per_due_date": "One Bill of Exchange per Due Date",
                            "per_due_date_per": "One Bill of Exch. per Due Date Per",
                        }
                        label = radio_map.get(create_boe)
                        if label: await _set_radio_by_label(webgui_frame, label)
                min_in = control_data.get("Minimum Amount for Incoming Payment")
                if min_in is None: min_in = target.get("Min_Incoming_Payment")
                if min_in is not None:
                    await _set_field_by_label_in_section(webgui_frame, "Minimum Amount for Incoming Payment", "Control Data", str(min_in))
                
                min_in_curr = control_data.get("Incoming Payment Currency")
                if min_in_curr:
                    await _set_field_by_label_in_section(webgui_frame, "Incoming Payment Currency", "Control Data", str(min_in_curr))

                min_out = control_data.get("Minimum Amount for Outgoing Payment")
                if min_out is None: min_out = target.get("Min_Outgoing_Payment")
                if min_out is not None:
                    await _set_field_by_label_in_section(webgui_frame, "Minimum Amount for Outgoing Payment", "Control Data", str(min_out))
                
                min_out_curr = control_data.get("Outgoing Payment Currency")
                if min_out_curr:
                    await _set_field_by_label_in_section(webgui_frame, "Outgoing Payment Currency", "Control Data", str(min_out_curr))

                # Checkboxes
                checkbox_map = {
                    "No Exchange Rate Differences": "No Exchange Rate Differences",
                    "No_Exchange_Rate_Diff": "No Exchange Rate Differences",
                    "No Exch.Rate Diffs. (Part Payments)": "No Exch.Rate Diffs. (Part Payments)",
                    "No_Exch_Rate_Diff_Part": "No Exch.Rate Diffs. (Part Payments)",
                    "Separate Payment for Each Ref.": "Separate Payment for Each Ref",
                    "Separate_Payment_Each_Ref": "Separate Payment for Each Ref",
                    "Bill/Exch Pymt": "Bill/Exch Pymt",
                    "Bill_Exch_Pymt": "Bill/Exch Pymt",
                    "Direct Debit Pre-Notifications": "Direct Debit Pre-Notifications",
                    "Direct_Debit_Pre_Notifications": "Direct Debit Pre-Notifications",
                }
                for json_key, label in checkbox_map.items():
                    val = control_data.get(json_key)
                    if val is not None:
                        await _set_checkbox_by_label(webgui_frame, label, bool(val))

                # --- Section: Specifications for SEPA Payments ---
                cred_id = sepa_data.get("Creditor Identification Number")
                if cred_id is None: cred_id = target.get("Creditor_ID_Number")
                if cred_id is not None:
                    await _set_field_by_label_in_section(webgui_frame, "Creditor Identification Number", "Specifications for SEPA Payments", str(cred_id))

                # ════════════════════════════════════════════════════════
                # DETAIL VIEW — Bill of Exchange Data (optional section)
                # ════════════════════════════════════════════════════════

                # Radio buttons: Create Bills of Exchange
                create_boe = target.get("Create_Bills_Exchange")
                if create_boe is not None:
                    radio_map = {
                        "per_invoice": "One Bill of Exchange per Invoice",
                        "per_due_date": "One Bill of Exchange per Due Date",
                        "per_due_date_per": "One Bill of Exch. per Due Date Per",
                    }
                    radio_label = radio_map.get(create_boe)
                    if radio_label:
                        await _set_radio_by_label(webgui_frame, radio_label)
                    else:
                        LOGGER.warning(
                            f"  Unknown Create_Bills_Exchange value: '{create_boe}'"
                        )

                # Bill of Exch.Due Date — Incoming Payments
                # Scroll down to make Bill of Exchange section visible
                await webgui_frame.evaluate("""() => {
                    const els = document.querySelectorAll('span, div, td');
                    for (const el of els) {
                        if ((el.textContent || '').includes('Bill of Exch.Due Date')) {
                            el.scrollIntoView({ behavior: 'instant', block: 'center' });
                            break;
                        }
                    }
                }""")
                await asyncio.sleep(0.5)

                incoming_boe_section = boe_data.get("Bill of Exch.Due Date/Bill of Exch.Pmnt Requests for Incoming Payments", {})
                latest_in = incoming_boe_section.get("Latest Due Date in Days")
                if latest_in is None: latest_in = target.get("Latest_Due_Date_Incoming")
                if latest_in is not None:
                    await _set_field_by_label_in_section(
                        webgui_frame,
                        "Latest Due Date in",
                        "Bill of Exch.Due Date/Bill of Exch.Pmnt Requests for Incoming Payments",
                        str(latest_in),
                    )

                demand_in = incoming_boe_section.get("Bill on Demand for Due Date up Until Days")
                if demand_in is None: demand_in = target.get("Bill_On_Demand_Due_Date")
                if demand_in is not None:
                    await _set_field_by_label_in_section(
                        webgui_frame,
                        "Bill on Demand for Due Date up Until",
                        "Bill of Exch.Due Date/Bill of Exch.Pmnt Requests for Incoming Payments",
                        str(demand_in),
                    )

                # Bill of Exchange Due Date — Outgoing Payments
                outgoing_boe_section = boe_data.get("Bill of Exchange Due Date for Outgoing Payments", {})
                earliest_out = outgoing_boe_section.get("Earliest Due Date in Days")
                if earliest_out is None: earliest_out = target.get("Earliest_Due_Date_Outgoing")
                if earliest_out is not None:
                    await _set_field_by_label_in_section(
                        webgui_frame,
                        "Earliest Due Date in",
                        "Bill of Exchange Due Date for Outgoing Payments",
                        str(earliest_out),
                    )

                latest_out = outgoing_boe_section.get("Latest Due Date in Days")
                if latest_out is None: latest_out = target.get("Latest_Due_Date_Outgoing")
                if latest_out is not None:
                    await _set_field_by_label_in_section(
                        webgui_frame,
                        "Latest Due Date in",
                        "Bill of Exchange Due Date for Outgoing Payments",
                        str(latest_out),
                    )

                LOGGER.info(f"  ✔ CoCd '{cocd}' detail edits complete.")

                # ════════════════════════════════════════════════════════
                # SAVE & NAVIGATE BACK
                # ════════════════════════════════════════════════════════
                LOGGER.info("  Saving changes...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page, timeout_ms=3000)
                await asyncio.sleep(1)

                # Go back to list view
                LOGGER.info("  Returning to list view (F3)...")
                await page.keyboard.press("F3")
                await asyncio.sleep(3)

                # Verify we're back on the list
                webgui_frame = await get_webgui_frame(page) or page
                try:
                    pos_visible = await webgui_frame.locator(
                        "text=/Position/i"
                    ).first.is_visible(timeout=3000)
                    if pos_visible:
                        LOGGER.info("  Back on list view.")
                    else:
                        LOGGER.warning(
                            "  Position button not visible — waiting longer."
                        )
                        await asyncio.sleep(3)
                except:
                    LOGGER.warning("  Could not verify list view state.")

            LOGGER.info("ALL TARGETS PROCESSED.")
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"Error in 101001 automation: {e}", exc_info=True)
        finally:
            if not page.is_closed():
                await browser.close()


async def Maintain_Additional_Parameters_102739(targets: list[dict]):
    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=FINSC_T001Z_N&CustomizingObject=C_T001Z_N&CustomizingObjectType=C"
        "&CustomizingProject=&CustomizingTransaction=S_ER9_11001055&Type=SSCUI"
    )
    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)
        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="domcontentloaded")
            await login(page, EMAIL, PASSWORD)
            await asyncio.sleep(5)
            if await check_and_abort_if_locked(page):
                return
            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("Frame not found!")
                return {"status": "error", "message": "Frame not found"}
            execution_results = []
            for target in targets:
                cocd = str(target.get("CoCd", "")).strip()
                params = target.get("Parameters", [])
                if not cocd or not params:
                    continue
                LOGGER.info(f"[102739] PROCESSING CoCd={cocd}")
                # ── Step 1: Find row and Click Checkbox ──
                row_info = None
                for scroll_attempt in range(5):
                    if scroll_attempt == 0:
                        try:
                            pos_bts = webgui_frame.locator("text=/Position/i")
                            count = await pos_bts.count()

                            p_btn = None
                            for i in range(count):
                                if await pos_bts.nth(i).is_visible():
                                    p_btn = pos_bts.nth(i)
                                    break
                            if p_btn:
                                await p_btn.click(force=True)
                                await asyncio.sleep(1.5)

                                dlg = webgui_frame.locator(
                                    "[role='dialog'], .sapUiWindow, .sapMDialog"
                                ).first
                                if await dlg.is_visible(timeout=3000):
                                    s_input = dlg.locator(
                                        "input[type='text'], .lsControl--lsEdit"
                                    ).first
                                    await s_input.click()
                                    await page.keyboard.press("Control+a")
                                    await page.keyboard.press("Backspace")
                                    await page.keyboard.type(cocd)
                                    await page.keyboard.press("Enter")
                                    await dlg.wait_for(state="hidden", timeout=5000)
                                    await asyncio.sleep(1.5)
                        except Exception as e:
                            LOGGER.warning(f"  Step 1 Position failed: {e}")

                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)
                    if row_info:
                        break
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(1)

                if not row_info:
                    LOGGER.error(f"  ✘ CoCd '{cocd}' not found. Skipping.")
                    execution_results.append(
                        {
                            "CoCd": cocd,
                            "Status": "Not Found",
                            "Message": "Company Code not visible in main list",
                        }
                    )
                    continue

                prefix = row_info["prefix"]
                r_idx = row_info["rowIdx"]

                # Checkbox is usually in the first column (index 0)
                chk_id = f"{prefix}[{r_idx},0]"
                LOGGER.info(f"  Selecting CoCd checkbox: {chk_id}")
                try:
                    chk = webgui_frame.locator(f"id={chk_id}")
                    if await chk.count() > 0:
                        await chk.first.click(force=True)
                    else:
                        await webgui_frame.evaluate(
                            f"() => document.getElementById('{chk_id}')?.click()"
                        )
                except Exception as e:
                    LOGGER.warning(
                        f"  First attempt to click checkbox failed: {e}. Trying suffix variant."
                    )
                    await webgui_frame.evaluate(
                        f"() => (document.getElementById('{chk_id}_c') || document.getElementById('{chk_id}'))?.click()"
                    )

                await asyncio.sleep(1)

                # ── Step 2: Navigate to Additional Data ──
                LOGGER.info("  Navigating to 'Additional Data' via sidebar...")
                nav_success = False
                for click_attempt in range(4):
                    try:
                        # 1. Ensure the folder is expanded if needed
                        cc_root = webgui_frame.locator(
                            "[role='treeitem']:has-text('Company Code')"
                        ).first
                        if await cc_root.is_visible(timeout=1000):
                            # Double click root to be sure it's expanded
                            await cc_root.dblclick(force=True)
                            await asyncio.sleep(1)

                        # 2. Try to find 'Additional Data' node
                        # SAP often uses 'text' or 'title' for these tree nodes
                        selectors = [
                            "[role='treeitem']:has-text('Additional Data')",
                            "[role='treeitem'][title*='Additional Data']",
                            "span:has-text('Additional Data')",
                            "text='Additional Data'",
                        ]

                        node = None
                        for sel in selectors:
                            candidate = webgui_frame.locator(sel).first
                            if await candidate.is_visible(timeout=1000):
                                node = candidate
                                break

                        if node:
                            await node.click(force=True)
                            await asyncio.sleep(0.5)
                            await node.dblclick(force=True)
                            await asyncio.sleep(3)

                            # Check for 'Par.type' which is the header of the target table
                            if await webgui_frame.locator(
                                "text='Par.type'"
                            ).first.is_visible(timeout=3000):
                                nav_success = True
                                break
                    except Exception as nav_e:
                        LOGGER.warning(
                            f"  Navigation attempt {click_attempt + 1} failed with error: {nav_e}"
                        )
                        await asyncio.sleep(1)

                if not nav_success:
                    LOGGER.error(
                        "  ✘ Could not confirm navigation to 'Additional Data' screen."
                    )
                    continue

                await asyncio.sleep(1)

                # ── Step 3: Maintain Parameters ──
                for p_item in params:
                    p_type = str(p_item["Type"]).strip()
                    p_val = str(p_item["Value"]).strip()

                    LOGGER.info(f"    Searching for parameter type: {p_type}")
                    search_success = False

                    for p_search_attempt in range(2):
                        try:
                            # Use text=/Position/i which works well on SAP lists
                            pos_bts = webgui_frame.locator("text=/Position/i")
                            count = await pos_bts.count()

                            p_btn = None
                            for i in range(count):
                                if await pos_bts.nth(i).is_visible():
                                    p_btn = pos_bts.nth(i)
                                    break

                            if p_btn:
                                await p_btn.click(force=True)
                                await asyncio.sleep(1.5)

                                # Handle the Position dialog generically
                                dlg = webgui_frame.locator(
                                    "[role='dialog'], .sapUiWindow, .sapMDialog"
                                ).first
                                if await dlg.is_visible(timeout=3000):
                                    s_input = dlg.locator(
                                        "input[type='text'], .lsControl--lsEdit"
                                    ).first
                                    await s_input.click()
                                    await page.keyboard.press("Control+a")
                                    await page.keyboard.press("Backspace")
                                    await page.keyboard.type(p_type)
                                    await page.keyboard.press("Enter")
                                    # Wait for dialog to disappear meaning search is done
                                    await dlg.wait_for(state="hidden", timeout=5000)
                                    await asyncio.sleep(1.5)
                                    search_success = True
                                    break
                        except Exception as e:
                            LOGGER.warning(
                                f"    Parameter search attempt {p_search_attempt + 1} failed: {e}"
                            )

                    if not search_success:
                        LOGGER.warning(
                            f"    ⚠ Position search for {p_type} failed. Attempting direct JS scan as fallback."
                        )

                    # STRICT Exact Match Logic for Par.type
                    set_res = await webgui_frame.evaluate(
                        """
                        (targetType) => {
                            const val = targetType.trim().toLowerCase();
                            const allElements = document.querySelectorAll('span, div, td, input');
                            let foundRow = null;
                            let prefix = null;

                            for (let el of allElements) {
                                // We check text content, or value if it's an input
                                const text = (el.tagName === 'INPUT' ? el.value : el.textContent).trim().toLowerCase();
                                if (text === val && el.id) {
                                    const m = el.id.match(/(.+)\\[(\\d+),/);
                                    if (m) {
                                        prefix = m[1];
                                        foundRow = m[2];
                                        break;
                                    }
                                }
                            }

                            if (foundRow !== null) {
                                // Now find the input field in this row
                                const selectors = `input[id^="${prefix}[${foundRow},"]`;
                                const inputs = document.querySelectorAll(selectors);

                                // Standard: parameter value is the rightmost input in the row or has type text
                                let targetInputId = null;
                                for (let i = inputs.length - 1; i >= 0; i--) {
                                    if (inputs[i].type === 'text' || inputs[i].className.includes('Edit')) {
                                        targetInputId = inputs[i].id;
                                        break;
                                    }
                                }

                                if (!targetInputId && inputs.length > 0) {
                                    targetInputId = inputs[inputs.length - 1].id;
                                }

                                if (targetInputId) {
                                    return { found: true, id: targetInputId, row: foundRow };
                                } else {
                                    // Fallback to static column guess
                                    return { found: true, id: `${prefix}[${foundRow},3]`, row: foundRow };
                                }
                            }
                            return { found: false };
                        }
                    """,
                        p_type,
                    )

                    if set_res and set_res.get("found"):
                        target_id = set_res["id"]
                        LOGGER.info(f"    Targeting input element: {target_id}")
                        try:
                            # Focus and click the element specifically
                            await webgui_frame.evaluate(
                                f"""(id) => {{
                                const el = document.getElementById(id);
                                if (el) {{
                                    el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                    el.focus();
                                    el.click();
                                }}
                            }}""",
                                target_id,
                            )
                            await asyncio.sleep(0.5)

                            # Clear using Ctrl+A and Delete, just like other working functions
                            await page.keyboard.press("Control+a")
                            await asyncio.sleep(0.1)
                            await page.keyboard.press("Delete")
                            await asyncio.sleep(0.1)

                            await page.keyboard.type(p_val, delay=50)
                            await asyncio.sleep(0.3)
                            # Using Tab is critical to trigger change events in SAP
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(1.0)

                        except Exception as e:
                            LOGGER.error(f"    Failed to set parameter value: {e}")
                    else:
                        LOGGER.error(
                            f"    ✘ Parameter type '{p_type}' not found in table."
                        )

                # ── Step 4: Save and handle validation errors ──
                LOGGER.info("  Saving changes...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page, timeout_ms=3000)

                # Check status bar for validation errors
                status = await get_status_bar_message(webgui_frame)
                if status and status["type"] == "error":
                    LOGGER.error(
                        f"  ❌ SAP Validation Error for CoCd {cocd}: {status['text']}"
                    )
                    execution_results.append(
                        {
                            "CoCd": cocd,
                            "Status": "Validation Error",
                            "Message": status["text"],
                        }
                    )

                    # Instead of reloading, cancel changes and return to 'Company Code'
                    LOGGER.info(
                        "  Navigating back to Company Code list ignoring validation..."
                    )
                    # Cancel shortcut F12 or repeatedly clicking 'Company Code'
                    sidebar_cc = webgui_frame.locator(
                        "[role='treeitem']:has-text('Company Code')"
                    ).first
                    if await sidebar_cc.is_visible(timeout=3000):
                        await sidebar_cc.click(force=True)
                        await asyncio.sleep(2)
                        await handle_sap_confirmation_dialogs(
                            page, timeout_ms=3000, default_action="No"
                        )
                    continue  # Skip to next target

                execution_results.append(
                    {
                        "CoCd": cocd,
                        "Status": "Success",
                        "Message": "Parameters maintained successfully",
                    }
                )

                # ── Step 5: Navigate back to Company Code folder ──
                LOGGER.info("  Returning to Company Code node...")
                try:
                    # Robust clicking of the Company Code tree node folder
                    sidebar_cc = webgui_frame.locator(
                        "[role='treeitem']:has-text('Company Code')"
                    ).first
                    if await sidebar_cc.is_visible(timeout=3000):
                        await sidebar_cc.click(force=True)
                        await asyncio.sleep(2)

                        # Verify we are on the Company Code screen
                        # If we see the 'Par.type' text, we are still in Additional Data. Let's F3 if so.
                        still_in_add = await webgui_frame.locator(
                            "text='Par.type'"
                        ).first.is_visible(timeout=2000)
                        if still_in_add:
                            LOGGER.info("  Still in Additional Data, trying F3...")
                            await page.keyboard.press("F3")
                            await asyncio.sleep(2)
                    else:
                        await page.keyboard.press("F3")
                        await asyncio.sleep(2)
                except Exception as e:
                    LOGGER.warning(f"  Error navigating back to Company Code: {e}")
                    await page.keyboard.press("F3")
                    await asyncio.sleep(2)

            LOGGER.info("✔ All company codes in batch processed.")
            await smart_logout(page)
            return {"status": "success", "results": execution_results}

        except Exception as e:
            LOGGER.error(f"Error in 102739 automation: {e}", exc_info=True)
            return {
                "status": "error",
                "message": str(e),
                "results": locals().get("execution_results", []),
            }
        finally:
            if "browser" in locals() and not page.is_closed():
                await browser.close()


async def Set_Up_All_Company_Codes_for_Payment_Transactions_101293(targets: list[dict]):
    """
    Set Up All Company Codes for Payment Transactions (Activity 101293).

    Opens the SAP SSCUI for maintaining payment transaction settings per company code.
    Double-clicks a company code row to enter its detail view, sets checkboxes, tolerance
    fields, and Special G/L Indicator text fields for Vendors and Customers, then saves.

    Parameters
    ----------
    targets : list[dict]
        Each dict represents one company code to configure. Keys:

        - "CoCd" (str, required): Company Code, e.g. "1810".
        - "Separate Payment per Business Area" (bool, optional): Checkbox. Default False.
        - "Pyt Meth Suppl" (bool, optional): Payment Method Supplement checkbox. Default False.
        - "Tolerance Days for Payable" (str, optional): Number of tolerance days, e.g. "10".
        - "Outgoing Pmnt with Cash Disc.From" (str, optional): Cash discount %, e.g. "10".
        - "Max.Cash Discount" (bool, optional): Checkbox. Default False.
        - "Vendors" (dict, optional): Contains two text fields for Vendor Special G/L Indicators.
            - "Sp. G/L Transactions to Be Paid" (str): One or more Special G/L Indicator
              codes concatenated together, e.g. "J" or "F1ABCDGHIJKLMOPSVWX".
            - "Sp. G/L Trans. for Exception List" (str): One or more Special G/L Indicator
              codes concatenated together, e.g. "V" or "I".
        - "Customers" (dict, optional): Contains two text fields for Customer Special G/L Indicators.
            - "Sp. G/L Transactions to Be Paid" (str): One or more Special G/L Indicator
              codes concatenated together, e.g. "T" or "1AGK".
            - "Sp. G/L Trans. for Exception List" (str): One or more Special G/L Indicator
              codes concatenated together, e.g. "A" or "T".

    Available Special G/L Indicators (Account Type K - Vendors)
    -----------------------------------------------------------
    These can be combined as a single string (e.g. "ABCJ") and typed directly
    into the text field. No need to open the selection popup.

        Code | Name
        ---- | ----
        1    | Guarantee Received
        A    | Down Payments, Current Assets
        B    | Down Payments, Financ'l Assets
        C    | Value-Dated Bank Transfer
        D    | Discounts
        F    | Down Payment Requests
        G    | Guarantee received
        H    | Security deposit
        I    | Down Payments, Intang. Assets
        J    | Advance Payment Request
        K    | AP Operating Costs
        L    | Down payment offset
        M    | Down Payments, Tangible Assets
        O    | Down Payments, Amortization
        P    | Payment request
        S    | Check/Bill of Exchange
        V    | Down Payments, Stocks
        W    | Rediscountable Bills of Exch.
        X    | Down Payment, Without Invoice

    Available Special G/L Indicators (Account Type D - Customers)
    -------------------------------------------------------------
    Same approach — combine codes as a single string.

        Code | Name
        ---- | ----
        1    | Guarantee Received
        A    | Down Payments, Current Assets
        G    | Guarantee received
        K    | AP Operating Costs
        T    | (Customer-specific indicator)

    Example payload
    ---------------
    >>> targets = [{
    ...     "CoCd": "1810",
    ...     "Separate Payment per Business Area": False,
    ...     "Pyt Meth Suppl": False,
    ...     "Tolerance Days for Payable": "10",
    ...     "Outgoing Pmnt with Cash Disc.From": "10",
    ...     "Max.Cash Discount": False,
    ...     "Vendors": {
    ...         "Sp. G/L Transactions to Be Paid": "F1ABCDGHIJKLMOPSVWX",
    ...         "Sp. G/L Trans. for Exception List": "I"
    ...     },
    ...     "Customers": {
    ...         "Sp. G/L Transactions to Be Paid": "1AGK",
    ...         "Sp. G/L Trans. for Exception List": "T"
    ...     }
    ... }]
    """

    url = (
        "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?"
        "IMGActivity=FIAPPY_V_T042&CustomizingObject=V_T042&"
        "CustomizingObjectType=V&CustomizingProject=&"
        "CustomizingTransaction=S_ALR_87100687&Type=SSCUI"
    )

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        # ── Local helpers ──
        # ── Robust Local JS Finders ──
        # These use Visual Bounding Box alignment (Y-axis matching) to pair labels and inputs
        # perfectly matching how they render on screen, immune to SAP WebGUI's complex table nesting.
        ROBUST_FIND_INPUT = """(labelText) => {
            const labelLower = labelText.toLowerCase().trim();
            function isValid(inp) {
                if (!inp || !inp.id) return false;
                const id = inp.id.toLowerCase();
                if (id.includes('toolbar') || id.includes('okcode')) return false;
                if (inp.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) return false;
                const type = (inp.type || '').toLowerCase();
                if (['checkbox', 'radio', 'hidden', 'submit', 'button'].includes(type)) return false;
                return true;
            }
            
            const labels = document.querySelectorAll('label, span, td, div');
            let bestMatch = null, minLen = 999;
            for (const lbl of labels) {
                const txt = (lbl.textContent || '').trim().toLowerCase();
                const cleanTxt = txt.replace(/[:*]/g, '').trim();
                if (cleanTxt.includes(labelLower) && txt.length < minLen && lbl.children.length <= 3) {
                    if (lbl.closest('[class*="urTbar"], [role="toolbar"], [role="button"], [class*="urBtn"], [class*="Button"]')) continue;
                    let rect = lbl.getBoundingClientRect();
                    if (rect.width === 0 || rect.height === 0) continue;
                    
                    bestMatch = lbl; 
                    minLen = txt.length;
                    if (cleanTxt === labelLower) minLen = -1; // Exact match priority
                }
            }
            if (!bestMatch) return null;
            
            // 1. htmlFor
            if (bestMatch.tagName === 'LABEL' && bestMatch.htmlFor) {
                const t = document.getElementById(bestMatch.htmlFor);
                if (t && isValid(t)) return { id: t.id };
            }
            
            // 2. Visual Alignment
            let rect = bestMatch.getBoundingClientRect();
            let labelCenterY = rect.top + rect.height / 2;
            let allInputs = Array.from(document.querySelectorAll('input, textarea')).filter(isValid);
            
            let bestInput = null;
            let minDistance = 999999;
            for (let inp of allInputs) {
                let inpRect = inp.getBoundingClientRect();
                if (inpRect.width === 0 || inpRect.height === 0) continue;
                let inpCenterY = inpRect.top + inpRect.height / 2;
                
                let yDiff = Math.abs(inpCenterY - labelCenterY);
                if (yDiff <= 15) { // Same visual row
                    let xDiff = inpRect.left - rect.right;
                    if (xDiff >= -150 && xDiff < minDistance) {
                        minDistance = xDiff;
                        bestInput = inp;
                    }
                }
            }
            if (bestInput) return { id: bestInput.id };
            
            // 3. Fallback: DOM walker
            let walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null, false);
            walker.currentNode = bestMatch;
            while (walker.nextNode()) {
                let node = walker.currentNode;
                if ((node.tagName === 'INPUT' || node.tagName === 'TEXTAREA') && isValid(node)) {
                    return { id: node.id };
                }
            }
            return null;
        }"""

        ROBUST_FIND_CHECKBOX = """(labelText) => {
            const labelLower = labelText.toLowerCase().trim();
            const labels = document.querySelectorAll('label, span, td, div');
            let bestMatch = null, minLen = 999;
            for (const lbl of labels) {
                const txt = (lbl.textContent || '').trim().toLowerCase();
                const cleanTxt = txt.replace(/[:*]/g, '').trim();
                if (cleanTxt.includes(labelLower) && txt.length < minLen && lbl.children.length <= 3) {
                    if (lbl.closest('[class*="urTbar"], [role="toolbar"], [role="button"], [class*="urBtn"], [class*="Button"]')) continue;
                    let r = lbl.getBoundingClientRect();
                    if (r.width === 0 || r.height === 0) continue;
                    
                    bestMatch = lbl; 
                    minLen = txt.length;
                    if (cleanTxt === labelLower) minLen = -1; // Exact match priority
                }
            }
            if (!bestMatch) return null;

            function getState(el) {
                if (!el) return null;
                if (el.tagName === 'INPUT' && el.type === 'checkbox') return { clickId: el.id, checked: el.checked };
                if (el.tagName === 'SPAN' && (el.className.includes('urCb') || el.getAttribute('role') === 'checkbox' || el.className.toLowerCase().includes('checkbox'))) {
                    const c = el.className.includes('urCbChk') || el.getAttribute('aria-checked') === 'true';
                    return { clickId: el.id, checked: c };
                }
                if (el.querySelector) {
                    const cb = el.querySelector('input[type="checkbox"], [role="checkbox"], span[class*="urCb"]');
                    if (cb) return getState(cb);
                }
                return null;
            }

            if (bestMatch.tagName === 'LABEL' && bestMatch.htmlFor) {
                const t = document.getElementById(bestMatch.htmlFor);
                const s = getState(t); if (s) return s;
            }
            
            // 2. Visual Alignment
            let rect = bestMatch.getBoundingClientRect();
            let labelCenterY = rect.top + rect.height / 2;
            let allCbs = Array.from(document.querySelectorAll('input[type="checkbox"], span[role="checkbox"], span[class*="urCb"]'));
            
            let bestCb = null;
            let minDistance = 999999;
            for (let cb of allCbs) {
                let cbRect = cb.getBoundingClientRect();
                if (cbRect.width === 0 || cbRect.height === 0) continue;
                let cbCenterY = cbRect.top + cbRect.height / 2;
                
                let yDiff = Math.abs(cbCenterY - labelCenterY);
                if (yDiff <= 15) { 
                    // Closest horizontally (could be left or right of label)
                    let dist = Math.min(Math.abs(cbRect.left - rect.right), Math.abs(cbRect.right - rect.left));
                    if (dist < minDistance) {
                        minDistance = dist;
                        bestCb = cb;
                    }
                }
            }
            if (bestCb) {
                const s = getState(bestCb);
                if (s) return s;
            }
            
            // 3. Fallback: DOM walker
            let walker = document.createTreeWalker(document.body, NodeFilter.SHOW_ELEMENT, null, false);
            walker.currentNode = bestMatch;
            while (walker.nextNode()) {
                let node = walker.currentNode;
                const s = getState(node);
                if (s) return s;
            }
            
            return null;
        }"""

        async def _set_field_by_label(wf, label_text, value):
            try:
                info = await wf.evaluate(ROBUST_FIND_INPUT, label_text)
                if not info:
                    LOGGER.warning(f"    Input for '{label_text}' not found. Skipping.")
                    return False
                input_id = info["id"]
                LOGGER.info(f"    Found input for '{label_text}' → #{input_id}")
                await wf.evaluate(
                    f"() => document.getElementById('{input_id}')?.scrollIntoView({{behavior:'instant', block:'center'}})"
                )
                await asyncio.sleep(0.3)
                loc = wf.locator(f"id={input_id}")
                if await loc.count() > 0:
                    await loc.first.click(force=True, timeout=3000)
                else:
                    await wf.evaluate(f"() => document.getElementById('{input_id}')?.click()")
                await asyncio.sleep(0.3)
                await page.keyboard.press("Control+a")
                await asyncio.sleep(0.1)
                await page.keyboard.press("Delete")
                await asyncio.sleep(0.1)
                await page.keyboard.type(str(value), delay=50)
                await asyncio.sleep(0.3)
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                LOGGER.info(f"    ✔ Set '{label_text}' = '{value}'")
                return True
            except Exception as e:
                LOGGER.warning(f"    Failed to set '{label_text}': {e}")
                return False

        async def _set_checkbox_by_label(wf, label_text, desired_state: bool):
            try:
                info = await wf.evaluate(ROBUST_FIND_CHECKBOX, label_text)
                if not info:
                    LOGGER.warning(f"    Checkbox for '{label_text}' not found. Skipping.")
                    return False
                click_id = info["clickId"]
                is_checked = info.get("checked", False)
                if is_checked != desired_state:
                    LOGGER.info(f"    Toggling '{label_text}' → {desired_state}")
                    await wf.evaluate(
                        f"() => document.getElementById('{click_id}')?.scrollIntoView({{behavior:'instant', block:'center'}})"
                    )
                    await asyncio.sleep(0.2)
                    await wf.locator(f"id={click_id}").first.click(force=True)
                    await asyncio.sleep(0.5)
                else:
                    LOGGER.info(f"    ✔ '{label_text}' already {'checked' if is_checked else 'unchecked'}")
                return True
            except:
                return False

        async def _scroll_to_section(wf, section_name):
            """Scroll a section heading into view so its fields become visible."""
            try:
                await wf.evaluate("""(sn) => {
                    const els = document.querySelectorAll('span, div, td, legend');
                    for (const h of els) {
                        const txt = (h.textContent || '').trim();
                        if (txt === sn && h.children.length <= 3) {
                            h.scrollIntoView({ behavior: 'instant', block: 'start' });
                            return true;
                        }
                    }
                    return false;
                }""", section_name)
                await asyncio.sleep(0.5)
            except:
                pass

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(5)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            for target in targets:
                cocd = str(target.get("CoCd", "1810"))
                LOGGER.info(f"PROCESSING CoCd={cocd}")

                # ── Step 1: Find Row or Create New Entry ──
                row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)

                if row_info:
                    prefix = row_info["prefix"]
                    r_idx = row_info["rowIdx"]
                    cell_id = f"{prefix}[{r_idx},2]"
                    LOGGER.info(f"  Found existing CoCd={cocd}, double-clicking...")
                    await webgui_frame.locator(f"id={cell_id}").first.dblclick(force=True)
                    await asyncio.sleep(4)
                    webgui_frame = await get_webgui_frame(page) or page
                else:
                    LOGGER.info(f"  CoCd '{cocd}' not found. Creating new entry...")
                    new_clicked = False
                    for sel in ["text=/New Entr/i", "[title*='New Entr']"]:
                        btn = webgui_frame.locator(sel).first
                        if await btn.is_visible(timeout=2000):
                            await btn.click(force=True)
                            new_clicked = True
                            break
                    if not new_clicked:
                        LOGGER.error(f"  ✘ 'New Entries' button not found. Skipping CoCd={cocd}.")
                        continue
                    await asyncio.sleep(3)
                    webgui_frame = await get_webgui_frame(page) or page

                    # Fill Company Code, Sending, and Paying using sequential Tab navigation.
                    # The label finder resolves both Sending & Paying to the same DOM element,
                    # so we use Tab to move between them like a human would.

                    # 1. Top-level Company Code
                    await _set_field_by_label(webgui_frame, "Company Code", cocd)

                    # 2. Click Sending company code, type, then Tab into Paying Company Code
                    sending_val = str(target.get("Sending Company Code", cocd))
                    paying_val = str(target.get("Paying Company Code", cocd))

                    info = await webgui_frame.evaluate(ROBUST_FIND_INPUT, "Sending company code")
                    if info:
                        inp_id = info["id"]
                        LOGGER.info(f"    Found 'Sending company code' → #{inp_id}")
                        await webgui_frame.evaluate(
                            f"() => document.getElementById('{inp_id}')?.scrollIntoView({{behavior:'instant', block:'center'}})"
                        )
                        await asyncio.sleep(0.3)
                        await webgui_frame.locator(f"id={inp_id}").first.click(force=True, timeout=3000)
                        await asyncio.sleep(0.3)
                        await page.keyboard.press("Control+a")
                        await asyncio.sleep(0.1)
                        await page.keyboard.press("Delete")
                        await page.keyboard.type(sending_val, delay=50)
                        await asyncio.sleep(0.3)
                        LOGGER.info(f"    ✔ Set 'Sending company code' = '{sending_val}'")

                        # Tab moves cursor to the next field: Paying Company Code
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.8)
                        await page.keyboard.press("Control+a")
                        await asyncio.sleep(0.1)
                        await page.keyboard.press("Delete")
                        await page.keyboard.type(paying_val, delay=50)
                        await asyncio.sleep(0.3)
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.5)
                        LOGGER.info(f"    ✔ Set 'Paying Company Code' = '{paying_val}'")
                    else:
                        LOGGER.warning("    Could not find 'Sending company code' input.")

                # ── Step 2: Control Data checkboxes ──
                LOGGER.info("  Filling Control Data...")
                await _set_checkbox_by_label(
                    webgui_frame,
                    "Separate Payment per Business Area",
                    target.get("Separate Payment per Business Area", False),
                )
                await _set_checkbox_by_label(
                    webgui_frame, "Pyt Meth Suppl", target.get("Pyt Meth Suppl", False)
                )

                # ── Step 3: Cash Discount and Tolerances ──
                LOGGER.info("  Filling Cash Discount and Tolerances...")
                await _scroll_to_section(webgui_frame, "Cash Discount and Tolerances")
                
                if target.get("Tolerance Days for Payable"):
                    await _set_field_by_label(webgui_frame, "Tolerance Days for Payable", target["Tolerance Days for Payable"])
                if target.get("Outgoing Pmnt with Cash Disc.From"):
                    await _set_field_by_label(webgui_frame, "Outgoing Pmnt with Cash Disc.From", target["Outgoing Pmnt with Cash Disc.From"])
                    
                await _set_checkbox_by_label(
                    webgui_frame, "Max.Cash Discount", target.get("Max.Cash Discount", False),
                )

                # ── Step 4: Vendors G/L fields ──
                v_data = target.get("Vendors", {})
                if v_data:
                    LOGGER.info("  Filling Vendors section...")
                    await _scroll_to_section(webgui_frame, "Vendors")
                    await asyncio.sleep(0.5)

                    # Use section-aware JS to find the correct input in Vendors (not Customers)
                    v_gl_paid = v_data.get("Sp. G/L Transactions to Be Paid")
                    if v_gl_paid is not None:
                        found = await webgui_frame.evaluate("""() => {
                            const labels = document.querySelectorAll('label, span, td');
                            let inVendors = false;
                            for (const lbl of labels) {
                                const txt = (lbl.textContent || '').trim();
                                if (txt === 'Vendors') { inVendors = true; continue; }
                                if (txt === 'Customers') break;
                                if (inVendors && txt.includes('Sp. G/L Transactions to Be Paid') && lbl.children.length <= 3) {
                                    let row = lbl.closest('tr') || lbl.parentElement;
                                    if (row) {
                                        const inputs = row.querySelectorAll('input[type="text"], input:not([type]), textarea');
                                        for (const inp of inputs) {
                                            if (inp.id && !inp.id.toLowerCase().includes('toolbar') && !inp.id.toLowerCase().includes('okcode')) {
                                                inp.scrollIntoView({ behavior: 'instant', block: 'center' });
                                                return { id: inp.id };
                                            }
                                        }
                                    }
                                }
                            }
                            return null;
                        }""")
                        if found:
                            inp_id = found["id"]
                            LOGGER.info(f"    Found Vendors GL To Be Paid → #{inp_id}")
                            await asyncio.sleep(0.3)
                            await webgui_frame.locator(f"id={inp_id}").first.click(force=True, timeout=3000)
                            await asyncio.sleep(0.3)
                            await page.keyboard.press("Control+a")
                            await asyncio.sleep(0.1)
                            await page.keyboard.press("Delete")
                            await page.keyboard.type(str(v_gl_paid), delay=50)
                            await asyncio.sleep(0.3)
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.5)
                            LOGGER.info(f"    ✔ Set Vendors GL To Be Paid = '{v_gl_paid}'")

                            # Tab moved focus to Vendors Exception List
                            v_gl_exc = v_data.get("Sp. G/L Trans. for Exception List")
                            if v_gl_exc is not None:
                                await page.keyboard.press("Control+a")
                                await asyncio.sleep(0.1)
                                await page.keyboard.press("Delete")
                                await page.keyboard.type(str(v_gl_exc), delay=50)
                                await asyncio.sleep(0.3)
                                await page.keyboard.press("Tab")
                                await asyncio.sleep(0.5)
                                LOGGER.info(f"    ✔ Set Vendors GL Exc. List = '{v_gl_exc}'")
                        else:
                            LOGGER.warning("    Could not find Vendors GL To Be Paid input.")

                # ── Step 5: Customers G/L fields ──
                c_data = target.get("Customers", {})
                if c_data:
                    LOGGER.info("  Filling Customers section...")
                    await _scroll_to_section(webgui_frame, "Customers")
                    await asyncio.sleep(0.5)

                    c_gl_paid = c_data.get("Sp. G/L Transactions to Be Paid")
                    if c_gl_paid is not None:
                        found = await webgui_frame.evaluate("""() => {
                            const labels = document.querySelectorAll('label, span, td');
                            let inCustomers = false;
                            for (const lbl of labels) {
                                const txt = (lbl.textContent || '').trim();
                                if (txt === 'Customers') { inCustomers = true; continue; }
                                if (inCustomers && txt.includes('Sp. G/L Transactions to Be Paid') && lbl.children.length <= 3) {
                                    let row = lbl.closest('tr') || lbl.parentElement;
                                    if (row) {
                                        const inputs = row.querySelectorAll('input[type="text"], input:not([type]), textarea');
                                        for (const inp of inputs) {
                                            if (inp.id && !inp.id.toLowerCase().includes('toolbar') && !inp.id.toLowerCase().includes('okcode')) {
                                                inp.scrollIntoView({ behavior: 'instant', block: 'center' });
                                                return { id: inp.id };
                                            }
                                        }
                                    }
                                }
                            }
                            return null;
                        }""")
                        if found:
                            inp_id = found["id"]
                            LOGGER.info(f"    Found Customers GL To Be Paid → #{inp_id}")
                            await asyncio.sleep(0.3)
                            await webgui_frame.locator(f"id={inp_id}").first.click(force=True, timeout=3000)
                            await asyncio.sleep(0.3)
                            await page.keyboard.press("Control+a")
                            await asyncio.sleep(0.1)
                            await page.keyboard.press("Delete")
                            await page.keyboard.type(str(c_gl_paid), delay=50)
                            await asyncio.sleep(0.3)
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.5)
                            LOGGER.info(f"    ✔ Set Customers GL To Be Paid = '{c_gl_paid}'")

                            c_gl_exc = c_data.get("Sp. G/L Trans. for Exception List")
                            if c_gl_exc is not None:
                                await page.keyboard.press("Control+a")
                                await asyncio.sleep(0.1)
                                await page.keyboard.press("Delete")
                                await page.keyboard.type(str(c_gl_exc), delay=50)
                                await asyncio.sleep(0.3)
                                await page.keyboard.press("Tab")
                                await asyncio.sleep(0.5)
                                LOGGER.info(f"    ✔ Set Customers GL Exc. List = '{c_gl_exc}'")
                        else:
                            LOGGER.warning("    Could not find Customers GL To Be Paid input.")

                # ── Step 6: Save ──
                LOGGER.info("  Saving changes...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page)

                status = await get_status_bar_message(webgui_frame)
                if status and status["type"] == "error":
                    LOGGER.error(f"  ❌ SAP Validation Error for CoCd {cocd}: {status['text']}")
                    LOGGER.error("  ⚠ Cannot save. Please fix the error manually and re-run.")
                    await smart_logout(page)
                    return {"status": "error", "CoCd": cocd, "message": status["text"]}

                LOGGER.info(f"  ✔ CoCd {cocd} saved successfully.")

                # ── Step 7: Back to list ──
                await page.keyboard.press("F3")
                await asyncio.sleep(3)
                webgui_frame = await get_webgui_frame(page) or page

            LOGGER.info("DONE")
            await smart_logout(page)
            return {"status": "success"}
        except Exception as e:
            LOGGER.error(f"Error in 101293: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        finally:
            if not page.is_closed():
                await browser.close()

async def Set_Up_Bank_Determination_for_Payment_Transactions_101045(
    targets: list[dict],
):
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FIAPPY_VC_BANKDET&CustomizingObject=VC_BANKDET&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ALR_87100691&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        try:
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                return

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("Frame not found!")
                return

            # Wait for any table row to appear
            try:
                await webgui_frame.locator("[lsdata]").first.wait_for(
                    state="visible", timeout=30000
                )
                LOGGER.info("Table data detected.")
            except:
                LOGGER.warning("Timeout waiting for table data. Proceeding anyway...")

            await asyncio.sleep(5)  # Final stabilization sleep

            for target in targets:
                cocd = str(target.get("Paying Company Code", "1810")).strip()
                LOGGER.info(f"--- Processing Paying Company Code: {cocd} ---")

                # ── Step 1: Select Paying Company Code ──
                row_info = None
                for scroll_attempt in range(5):
                    LOGGER.info(f"  Search attempt {scroll_attempt + 1} for {cocd}...")
                    row_info = await webgui_frame.evaluate(JS_FIND_ROW_BY_COCD, cocd)
                    if row_info:
                        LOGGER.info(f"  ✔ Found {cocd} at row {row_info['rowIdx']}")
                        break

                    if scroll_attempt == 0:
                        try:
                            LOGGER.info(f"  Attempting Position to find {cocd}...")
                            pos_bts = webgui_frame.locator("text=/Position/i")
                            count = await pos_bts.count()
                            p_btn = None
                            for i in range(count):
                                if await pos_bts.nth(i).is_visible():
                                    p_btn = pos_bts.nth(i)
                                    break

                            if p_btn:
                                await p_btn.click(force=True)
                                await asyncio.sleep(2)
                                await page.keyboard.type(cocd)
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(3)
                            else:
                                # Fallback scroll if Position not found
                                await page.keyboard.press("PageDown")
                                await asyncio.sleep(2)
                        except:
                            pass
                    else:
                        await page.keyboard.press("PageDown")
                        await asyncio.sleep(2)

                if not row_info:
                    LOGGER.error(
                        f"  ✘ Paying Company Code '{cocd}' not found. Skipping."
                    )
                    continue

                # Highlight row by clicking the selection checkbox (Column 1)
                prefix = row_info["prefix"]
                r_idx = row_info["rowIdx"]
                c_idx = int(row_info["colIdx"])
                chk_id = f"{prefix}[{r_idx},{c_idx - 1}]"

                LOGGER.info(f"  Attempting to check checkbox for {cocd} at {chk_id}...")
                await webgui_frame.evaluate(
                    f"""(id) => {{
                    const el = document.getElementById(id);
                    if (el) {{
                        el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        el.click();
                        // Verify checkmark
                        const inp = el.querySelector('input[type="checkbox"]');
                        if (inp && !inp.checked) inp.click();
                    }}
                }}""",
                    chk_id,
                )
                await asyncio.sleep(1)

                # Press Enter to 'enter' the detail mode for this company code
                LOGGER.info("  Pressing Enter to enter detail view...")
                await page.keyboard.press("Enter")
                await asyncio.sleep(2)

                # ── Step 2: Process Sections ──
                section_mapping = [
                    ("Ranking Order", target.get("Ranking Order", [])),
                    ("Bank Accounts", target.get("Bank Accounts", [])),
                    (
                        "Bank Accounts (Enhanced)",
                        target.get("Bank Accounts (Enhanced)", []),
                    ),
                    ("Value Date", target.get("Value Date", [])),
                    ("Expenses/Charges", target.get("Expenses/Charges", [])),
                ]

                for section_name, rows in section_mapping:
                    if not rows:
                        continue
                    if isinstance(rows, dict) and not isinstance(rows, list):
                        rows = [rows]

                    LOGGER.info(f"  📂 Navigating to section: {section_name}")

                    nav_success = False
                    for nav_attempt in range(3):
                        try:
                            webgui_frame = await get_webgui_frame(page) or webgui_frame

                            # Try to navigate by clicking sidebar
                            sel = f"text='{section_name}'"
                            node = webgui_frame.locator(sel).first
                            if await node.is_visible(timeout=1000):
                                await node.click(force=True)
                                await asyncio.sleep(0.5)
                                await node.dblclick(force=True)
                                await asyncio.sleep(3)

                            # Check if we are in the right view (Header check)
                            # Looking for title specifically in the main content area
                            header = webgui_frame.locator(
                                f"text='{section_name}'"
                            ).first
                            if await header.is_visible(timeout=2000):
                                nav_success = True
                                break

                            # Fallback: Press Enter again
                            if nav_attempt == 1:
                                await page.keyboard.press("Enter")
                                await asyncio.sleep(2)
                        except Exception as e:
                            LOGGER.warning(
                                f"    Nav attempt {nav_attempt + 1} failed: {e}"
                            )
                            await page.keyboard.press("F3")
                            await asyncio.sleep(1.5)

                    if not nav_success:
                        LOGGER.warning(
                            f"  ⚠️ Navigation to '{section_name}' failed. Skipping section."
                        )
                        continue

                    # Fill Table via New Entries
                    LOGGER.info(f"    ➕ Adding entries for {section_name}...")

                    # Step 1: Click "New Entries" once for this section
                    clicked = False
                    try:
                        await webgui_frame.locator(
                            ".lsToolbar, [role='toolbar']"
                        ).first.evaluate(
                            "el => el.scrollIntoView({behavior:'instant', block:'center'})"
                        )
                    except:
                        pass

                    new_selectors = [
                        "text=/New Entr/i",
                        "[title*='New Entr']",
                        "[id*='btn']:has-text('New')",
                    ]
                    for sel in new_selectors:
                        btn = webgui_frame.locator(sel).first
                        if await btn.is_visible(timeout=1000):
                            await btn.click(force=True)
                            clicked = True
                            break

                    if not clicked:
                        # Toolbar Menu fallback
                        menu_sel = ".lsToolbar [title*='Menu'], .lsToolbar [id*='mnu']"
                        menu_btn = webgui_frame.locator(menu_sel).first
                        if await menu_btn.is_visible(timeout=1000):
                            await menu_btn.click(force=True)
                            await asyncio.sleep(0.5)
                            await page.locator("text=/New Entr/i").last.click(
                                force=True
                            )
                            clicked = True

                    if not clicked:
                        await page.keyboard.press("F5")
                        await asyncio.sleep(2)
                    else:
                        await asyncio.sleep(2)

                    # Step 2: Fill all rows
                    for r_idx_row, row_data in enumerate(rows):
                        try:
                            LOGGER.info(f"    Row {r_idx_row + 1}: Entering data...")

                            # Navigate to the start of the current row if not the first row
                            if r_idx_row > 0:
                                await page.keyboard.press("ArrowDown")
                                await asyncio.sleep(0.5)
                                await page.keyboard.press("Home")
                                await asyncio.sleep(0.5)

                            for f_idx, val in enumerate(row_data.values()):
                                # Tab to next field within the row
                                if f_idx > 0:
                                    await page.keyboard.press("Tab")
                                    await asyncio.sleep(0.2)

                                if val:
                                    await page.keyboard.press("Control+a")
                                    await page.keyboard.press("Backspace")
                                    await page.keyboard.type(str(val), delay=50)

                            # Press Enter to validate the current row before moving to next
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(1)
                        except Exception as row_e:
                            LOGGER.error(f"    Error in row {r_idx_row + 1}: {row_e}")

                    # Save section
                    await page.keyboard.press("Control+s")
                    await asyncio.sleep(2)
                    await handle_sap_confirmation_dialogs(page)

                    # ── Check bottom-left status bar for SAP errors ──
                    save_status = await get_status_bar_message(webgui_frame)
                    if save_status and save_status.get("type") == "error":
                        err_text = save_status.get("text", "Unknown SAP error after save")
                        LOGGER.error(
                            f"  ❌ SAP save error for CoCd={cocd}: {err_text}"
                        )
                        raise RuntimeError(
                            f"SAP save rejected for CoCd={cocd}: {err_text}"
                        )

                    # Back to Company Code list
                    await page.keyboard.press("F3")
                    await asyncio.sleep(2)

                LOGGER.info(f"  ✔ CoCd {cocd} complete.")

            LOGGER.info("DONE. EXITING.")
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"Task 101045 Error: {e}", exc_info=True)
        finally:
            if "browser" in locals() and not page.is_closed():
                await browser.close()


async def Define_Account_Determination_for_Bank_Clearing_Accounts_102803(
    targets: list[dict],
):
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFFIBAT042Y&CustomizingObject=V_T042Y&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87001860&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="domcontentloaded")
            await login(page, EMAIL, PASSWORD)
            await asyncio.sleep(5)

            if await check_and_abort_if_locked(page):
                return

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("Frame not found!")
                return

            for target in targets:
                cocd = str(target.get("Paying Company Code", "1810")).strip()
                entries = target.get("Entries", [])
                LOGGER.info(f"[102803] PROCESSING CoCd={cocd}")

                # ── Step 1: Handle Determine Work Area Dialog ──
                try:
                    # Wait for the dialog to appear
                    dlg = webgui_frame.locator(
                        "[role='dialog'], .sapUiWindow, .sapMDialog"
                    ).first
                    await dlg.wait_for(state="visible", timeout=10000)

                    # Type CoCd and press Enter
                    await page.keyboard.type(cocd)
                    await asyncio.sleep(0.5)
                    await page.keyboard.press("Enter")
                    await asyncio.sleep(3)
                except Exception as e:
                    LOGGER.warning(
                        f"  Work Area dialog handling failed or not found: {e}"
                    )

                # Check for lock once more before editing
                if await check_and_abort_if_locked(page):
                    return

                # ── Step 2: Click 'New Entries' ──
                LOGGER.info("  Clicking 'New Entries'...")
                clicked = False
                new_selectors = [
                    "text=/New Entr/i",
                    "[title*='New Entr']",
                    "[id*='btn']:has-text('New')",
                ]
                for sel in new_selectors:
                    btn = webgui_frame.locator(sel).first
                    if await btn.is_visible(timeout=2000):
                        await btn.click(force=True)
                        clicked = True
                        break

                if not clicked:
                    await page.keyboard.press("F5")

                await asyncio.sleep(2)
                # ── Step 3: Fill Rows with Robust Mapping ──
                # Dynamically detect if the grid starts at index 0 or 1
                base_row = await webgui_frame.evaluate("""() => {
                    const inputs = Array.from(document.querySelectorAll('input[id*="["]'));
                    if (inputs.length === 0) return 0;
                    const indices = inputs.map(i => {
                        const m = i.id.match(/\\[(\\d+),/);
                        return m ? parseInt(m[1]) : 999;
                    });
                    return Math.min(...indices);
                }""")
                LOGGER.info(f"    Table base row index: {base_row}")

                for r_idx, row_data in enumerate(entries):
                    try:
                        LOGGER.info(f"    Row {r_idx + 1}: Entering data...")
                        target_prefix = await webgui_frame.evaluate("""() => {
                            const anyInp = document.querySelector('input[id*="["]');
                            return anyInp && anyInp.id ? anyInp.id.split('[')[0] : null;
                        }""")

                        if not target_prefix:
                            LOGGER.error(
                                "    ✘ Could not determine table prefix. Skipping row."
                            )
                            continue

                        # Anchor to the start of the specified row by ID for every entry
                        target_id = f"{target_prefix}[{base_row + r_idx},1]"
                        await webgui_frame.evaluate(
                            f"""(id) => {{
                            const el = document.getElementById(id);
                            if (el) {{
                                el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                                el.focus(); el.click();
                            }}
                        }}""",
                            target_id,
                        )
                        await asyncio.sleep(0.5)

                        for f_idx, (key, val) in enumerate(row_data.items()):
                            if val:
                                await page.keyboard.press("Control+a")
                                await page.keyboard.press("Backspace")
                                await page.keyboard.type(str(val), delay=10)

                            # Move to the next field within the row
                            if f_idx < len(row_data) - 1:
                                await page.keyboard.press("Tab")
                                await asyncio.sleep(0.1)

                        # Finalize the row with Enter and wait for validation
                        await page.keyboard.press("Enter")
                        await asyncio.sleep(2.0)
                        # Press Home to return to the first column (House Bk) for the next row
                        await page.keyboard.press("Home")
                        await asyncio.sleep(0.5)

                    except Exception as row_e:
                        LOGGER.error(f"    Error in row {r_idx + 1}: {row_e}")

                # ── Step 4: Save ──
                LOGGER.info("  Saving changes...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page)

                # ── Check bottom-left status bar for SAP errors ──
                status = await get_status_bar_message(webgui_frame)
                if status and status.get("type") == "error":
                    err_text = status.get("text", "Unknown SAP error after save")
                    LOGGER.error(f"  ❌ SAP save error for CoCd={cocd}: {err_text}")
                    raise RuntimeError(
                        f"SAP save rejected for CoCd={cocd}: {err_text}"
                    )

                # Move back to start if there are more targets
                await page.keyboard.press("F3")
                await asyncio.sleep(2)

            LOGGER.info("✔ Task 102803 complete.")
            await smart_logout(page)

        except Exception as e:
            LOGGER.error(f"Error in 102803 automation: {e}", exc_info=True)
        finally:
            if "browser" in locals() and not page.is_closed():
                await browser.close()


async def Define_Clearing_Accounts_for_Receiving_Bank_for_Account_Transfer_102802(
    targets: list[dict],
):
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_CFFIBAT018V&CustomizingObject=V_T018V&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87001755&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)

            if await check_and_abort_if_locked(page):
                return

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WebGUI frame not found!")
                return

            await asyncio.sleep(5)  # Stabilization

            # Step 1: Click New Entries once to enter batch mode
            LOGGER.info("Navigating to 'New Entries'...")
            new_selectors = [
                "button[title*='New Entr']",
                ".lsToolbar--base [role='button']:has-text('New')",
                "[id*='btn']:has-text('New')",
            ]

            clicked = False
            for sel in new_selectors:
                btn = webgui_frame.locator(sel).first
                if await btn.is_visible(timeout=2000):
                    await btn.click(force=True)
                    clicked = True
                    break

            if not clicked:
                await page.keyboard.press("F5")

            await asyncio.sleep(4)
            webgui_frame = await get_webgui_frame(page) or webgui_frame

            # Detect the first empty row index
            base_row = await webgui_frame.evaluate("""() => {
                const inputs = Array.from(document.querySelectorAll('input[id*="["]'));
                if (inputs.length === 0) return 0;
                return Math.min(...inputs.map(i => parseInt(i.id.match(/\\[(\\d+),/)?.[1] || 999)));
            }""")
            LOGGER.info(f"Targeting grid starting at row index: {base_row}")

            for r_idx, target in enumerate(targets):
                cocode = str(target.get("CoCode", ""))
                LOGGER.info(f"--- Filling Entry {r_idx + 1} (CoCode: {cocode}) ---")

                # Anchor to the first field of the current row (Column index 1: CoCode)
                target_prefix = await webgui_frame.evaluate("""() => {
                    const anyInp = document.querySelector('input[id*="["]');
                    return anyInp && anyInp.id ? anyInp.id.split('[')[0] : null;
                }""")

                target_id = f"{target_prefix}[{base_row + r_idx},1]"
                await webgui_frame.evaluate(
                    f"""(id) => {{
                    const el = document.getElementById(id);
                    if (el) {{
                        el.scrollIntoView({{ behavior: 'instant', block: 'center' }});
                        el.focus(); el.click();
                    }}
                }}""",
                    target_id,
                )
                await asyncio.sleep(0.5)

                fields = [
                    target.get("CoCode"),
                    target.get("House Bk"),
                    target.get("Cntry/Reg."),
                    target.get("Payt Meth."),
                    target.get("Currency"),
                    target.get("Account ID"),
                    target.get("Clrg Acct"),
                ]

                for f_idx, val in enumerate(fields):
                    if f_idx > 0:
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.1)

                    if val:
                        await page.keyboard.press("Control+a")
                        await page.keyboard.press("Backspace")
                        await page.keyboard.type(str(val), delay=10)

                # Press Enter to validate the row
                await page.keyboard.press("Enter")
                await asyncio.sleep(1.5)
                LOGGER.info(f"✔ Entry {r_idx + 1} for {cocode} filled.")

            # Final Save
            LOGGER.info("ATTEMPTING TO SAVE CHANGES...")
            await page.keyboard.press("Control+s")
            await asyncio.sleep(3)
            # Use specific confirmation handler if execute_save_flow is too complex
            await handle_sap_confirmation_dialogs(page)

            # ── Check bottom-left status bar for SAP errors ──
            save_status = await get_status_bar_message(webgui_frame)
            if save_status and save_status.get("type") == "error":
                err_text = save_status.get("text", "Unknown SAP error after save")
                LOGGER.error(f"  ❌ SAP save error: {err_text}")
                raise RuntimeError(f"SAP save rejected: {err_text}")

            LOGGER.info("DONE.")

        except Exception as e:
            LOGGER.error(f"Automation error: {e}")
        finally:
            await asyncio.sleep(5)
            await browser.close()


async def Maintenance_of_Company_Code_Data_for_a_Payment_Method_101044(
    targets: list[dict],
):
    """
    SSCUI 101044 – Maintenance of Company Code Data for a Payment Method.
    Uses Tab-based sequential navigation (proven reliable in SAP WebGUI).

    Tab order on main screen (from Minimum Amount):
      1. Minimum Amount  (text)
      2. Maximum Amount  (text)
      3. Distrib. Amount (text)
      4. Single Payment for Marked Item (checkbox)
      5. Payment per Due Day (checkbox)
      6. Extended Individual Payment (checkbox)
      7. Foreign Business Partner Allowed (checkbox)
      8. Foreign Currency Allowed (checkbox)
      9. Bank Abroad Allowed (checkbox)
      10-12. Bank Selection Control radio group

    After clicking "Form Data" button (expands inline):
      13. Form for Payment Medium (dropdown) – skip via Tab
      14. Form name (text)
      15-18. Drawer lines 1-4 (text)
      19. Correspondence (text)
      20. Line Items (text)
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FIAPPY_V_T042E&CustomizingObject=V_T042E&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87100690&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        # ── Proven helpers from FIN_configuration.py ──

        async def _type_field(value):
            """Select all → delete → type new value."""
            await page.keyboard.press("Control+a")
            await asyncio.sleep(0.2)
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.2)
            await page.keyboard.type(str(value), delay=50)
            await asyncio.sleep(0.5)

        async def _tab_and_type(field_name, value):
            """Tab to next field and type a value. If value is empty, just Tab past it."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.5)
            if value:
                await _type_field(value)
                LOGGER.info(f"    ✔ '{field_name}' = '{value}'")
            else:
                LOGGER.info(f"    ⏭ '{field_name}' (skipped, empty)")

        async def _set_checkbox_tab(field_name, target_state, wf):
            """Tab to checkbox, detect state via lsdata, toggle with Space if needed."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.8)

            if target_state is None:
                LOGGER.info(f"    ⏭ Checkbox '{field_name}': skipped")
                return

            # Detect checkbox state from the focused element using lsdata
            JS_CHECKBOX_DETECT = """
            () => {
                const el = document.activeElement;
                if (!el) return null;

                // Strategy 1: lsdata on element or parent
                const checkLsdata = (n) => {
                    if (!n) return null;
                    const raw = n.getAttribute('lsdata');
                    if (raw) {
                        try {
                            const d = JSON.parse(raw.replace(/'/g, '"'));
                            if (d['1'] === true || d['1'] === '1' || d['1'] === 1) return true;
                            if (d['1'] === false || d['1'] === '0' || d['1'] === 0 || d['1'] === '') return false;
                        } catch(e) {}
                    }
                    return null;
                };
                let s = checkLsdata(el);
                if (s !== null) return s;
                s = checkLsdata(el.parentElement);
                if (s !== null) return s;

                // Strategy 2: native checkbox
                if (el.tagName === 'INPUT' && el.type === 'checkbox') return el.checked;
                const inp = el.querySelector('input[type="checkbox"]');
                if (inp) return inp.checked;

                // Strategy 3: aria-checked
                const aria = el.getAttribute('aria-checked');
                if (aria === 'true') return true;
                if (aria === 'false') return false;

                // Strategy 4: CSS class
                const cls = (el.className || '') + ' ' + (el.parentElement?.className || '');
                if (cls.includes('Chk')) return true;
                if (cls.includes('Uchk')) return false;

                return null;
            }
            """
            try:
                state = await wf.evaluate(JS_CHECKBOX_DETECT)
                LOGGER.info(
                    f"    Checkbox '{field_name}': detected={state}, target={target_state}"
                )

                if state is not None:
                    if state != target_state:
                        LOGGER.info(f"      → Toggling '{field_name}' (Space)")
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.5)
                    else:
                        LOGGER.info(f"      → Already correct")
                else:
                    LOGGER.warning(
                        f"      ! Detection failed — fallback: toggle if target=True"
                    )
                    if target_state:
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.5)
            except Exception as e:
                LOGGER.warning(f"      ! Error: {e} — fallback toggle")
                if target_state:
                    await page.keyboard.press("Space")
                    await asyncio.sleep(0.5)

        async def _click_first_input(wf, label_text):
            """Click the first input field near a label to anchor the Tab sequence."""
            found = await wf.evaluate(
                """(labelText) => {
                const allElements = Array.from(document.querySelectorAll('span, label, td'));
                for (const el of allElements) {
                    const t = el.textContent.trim();
                    if (t.includes(labelText) && t.length < labelText.length + 15) {
                        let parent = el.parentElement;
                        for (let i = 0; i < 6; i++) {
                            if (!parent) break;
                            const inp = parent.querySelector('input[type="text"], input:not([type]), input[type="number"]');
                            if (inp && !inp.disabled && !inp.readOnly) {
                                inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                inp.focus();
                                inp.click();
                                return true;
                            }
                            parent = parent.parentElement;
                        }
                    }
                }
                return false;
            }""",
                label_text,
            )
            return found

        async def _click_sap_button(wf, button_text):
            """Click an SAP WebGUI section button by text using Playwright native click."""
            before_count = await wf.evaluate(
                "() => document.querySelectorAll('input').length"
            )

            success = False
            # Strategy 1: Use Playwright locator with exact-ish text match
            try:
                btn = wf.get_by_text(button_text, exact=False).first
                await btn.scroll_into_view_if_needed(timeout=3000)
                await btn.click(timeout=5000)
                LOGGER.info(
                    f"    Button '{button_text}' clicked via Playwright get_by_text"
                )
                success = True
            except Exception as e1:
                LOGGER.warning(f"    Playwright get_by_text failed: {e1}")
                # Strategy 2: Playwright regex locator
                try:
                    escaped = button_text.replace(" ", ".*")
                    btn = wf.locator(f"text=/{escaped}/i").first
                    await btn.scroll_into_view_if_needed(timeout=3000)
                    await btn.click(timeout=5000)
                    LOGGER.info(
                        f"    Button '{button_text}' clicked via Playwright regex"
                    )
                    success = True
                except Exception as e2:
                    LOGGER.warning(f"    Playwright regex failed: {e2}")
                    # Strategy 3: JS to find and scroll into view, then Playwright click by coordinates
                    try:
                        coords = await wf.evaluate(
                            """(btnText) => {
                            const allEls = document.querySelectorAll('span, div, a, td, button, [role="button"]');
                            for (const el of allEls) {
                                const t = (el.textContent || '').trim();
                                if (t.toLowerCase().includes(btnText.toLowerCase()) && t.length < btnText.length + 15) {
                                    el.scrollIntoView({behavior: 'instant', block: 'center'});
                                    const rect = el.getBoundingClientRect();
                                    return {x: rect.x + rect.width/2, y: rect.y + rect.height/2};
                                }
                            }
                            return null;
                        }""",
                            button_text,
                        )
                        if coords:
                            # Click at the coordinates within the frame
                            await wf.click(
                                f"text=/{button_text}/i", force=True, timeout=5000
                            )
                            LOGGER.info(
                                f"    Button '{button_text}' clicked via JS+force"
                            )
                            success = True
                        else:
                            LOGGER.error(
                                f"    Could not find '{button_text}' button at all"
                            )
                    except Exception as e3:
                        LOGGER.error(f"    All click strategies failed: {e3}")

            if not success:
                return False

            await asyncio.sleep(4)  # SAP needs time for server roundtrip

            # Verify expansion
            after_count = await wf.evaluate(
                "() => document.querySelectorAll('input').length"
            )
            LOGGER.info(f"    Inputs: before={before_count}, after={after_count}")

            if after_count <= before_count:
                LOGGER.warning(f"    Section may not have expanded! Retrying click...")
                try:
                    btn = wf.get_by_text(button_text, exact=False).first
                    await btn.click(timeout=5000)
                    await asyncio.sleep(4)
                    after_count2 = await wf.evaluate(
                        "() => document.querySelectorAll('input').length"
                    )
                    LOGGER.info(f"    Inputs after retry: {after_count2}")
                except Exception:
                    pass

            return True

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                return

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                return

            await asyncio.sleep(5)

            for target in targets:
                cocd = str(target.get("CoCode", "1810"))
                pyt_meth = str(target.get("Payt Meth.", "C"))
                data = target.get("Data", {})
                LOGGER.info(f"--- Processing CoCd: {cocd}, Payt Meth: {pyt_meth} ---")

                # ── Step 1: Position search ──
                try:
                    pos_btn = webgui_frame.locator("text=/Position/i").first
                    await pos_btn.scroll_into_view_if_needed()
                    await pos_btn.click(force=True, timeout=10000)
                except:
                    LOGGER.warning(
                        "  Position button not found via text, trying button search..."
                    )
                    await webgui_frame.evaluate("""() => {
                        const btns = document.querySelectorAll('button, span, div');
                        for (const b of btns) {
                            if ((b.textContent || '').trim().match(/^Position/i)) { b.click(); return; }
                        }
                    }""")
                await asyncio.sleep(2)

                await page.keyboard.type(cocd, delay=30)
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.3)
                await page.keyboard.type(pyt_meth, delay=30)
                await page.keyboard.press("Enter")
                await asyncio.sleep(3)

                # ── Step 2: Open detail (Update or New Entry) ──
                is_new_entry = False
                row_found = await webgui_frame.evaluate(
                    """([c, p]) => {
                    const rows = document.querySelectorAll('tr, div[role="row"]');
                    for (const row of rows) {
                        const t = row.textContent || "";
                        if (t.includes(c) && t.includes(p)) {
                            row.scrollIntoView({behavior: 'instant', block: 'center'});
                            return true;
                        }
                    }
                    return false;
                }""",
                    [cocd, pyt_meth],
                )

                if row_found:
                    LOGGER.info(
                        f"  Row {cocd}/{pyt_meth} found. Opening detail view..."
                    )
                    row_sel = webgui_frame.locator(f"text='{cocd}'").first
                    # Try to be more specific if possible, but dblclick is usually fine on the CoCd cell
                    await row_sel.dblclick(force=True)
                    await asyncio.sleep(5)
                else:
                    LOGGER.info(
                        f"  Row {cocd}/{pyt_meth} NOT found. Clicking 'New Entries'..."
                    )
                    await _click_sap_button(webgui_frame, "New Entries")
                    await asyncio.sleep(4)
                    is_new_entry = True

                LOGGER.info("  Detail view ready.")

                # Re-acquire frame after navigation
                webgui_frame = await get_webgui_frame(page) or webgui_frame
                await asyncio.sleep(2)

                # If New Entry, we must fill the keys first
                if is_new_entry:
                    LOGGER.info(f"  [New Entry] Filling CoCode and Payt Method...")
                    # Anchor on the first field
                    clicked = await _click_first_input(webgui_frame, "Paying Co. Code")
                    if not clicked:
                        LOGGER.warning(
                            "    Could not click 'Paying Co. Code'! Trying fallback Tab."
                        )
                        await page.keyboard.press("Tab")

                    await asyncio.sleep(0.5)
                    await _type_field(cocd)  # Paying CoCode

                    await page.keyboard.press("Tab")
                    await asyncio.sleep(0.5)
                    await _type_field(pyt_meth)  # Payment Method

                    await page.keyboard.press("Enter")  # Critical validation
                    await asyncio.sleep(4)
                    await handle_sap_confirmation_dialogs(page)
                    await asyncio.sleep(2)

                    # Now anchor on 'Minimum Amount' to continue standard flow
                    await _click_first_input(webgui_frame, "Minimum Amount")
                    await asyncio.sleep(0.5)

                # ═══════════════════════════════════════════════
                #  MAIN SCREEN — Tab-based sequential fill
                # ═══════════════════════════════════════════════

                amt = data.get("Amount Limits", {})
                grp = data.get("Grouping of Items", {})
                fgn = data.get("Foreign Payments", {})
                bsc = data.get("Bank Selection Control", {})

                # Click into first field: "Minimum Amount"
                LOGGER.info("  [Main] Clicking into Minimum Amount field...")
                clicked = await _click_first_input(webgui_frame, "Minimum Amount")
                if not clicked:
                    LOGGER.warning(
                        "  Could not click Minimum Amount! Trying by Tab from page start."
                    )
                    # Fallback: press Tab multiple times from the top
                    for _ in range(5):
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.3)
                else:
                    await asyncio.sleep(0.5)

                # Field 1: Minimum Amount (we're already focused on it)
                val = amt.get("Minimum Amount", "")
                if val:
                    await _type_field(val)
                    LOGGER.info(f"    ✔ 'Minimum Amount' = '{val}'")
                else:
                    LOGGER.info(f"    ⏭ 'Minimum Amount' (skipped)")

                # Field 2: Maximum Amount
                await _tab_and_type("Maximum Amount", amt.get("Maximum Amount", ""))

                # Field 3: Distrib. Amount
                await _tab_and_type("Distrib. Amount", amt.get("Distrib. Amount", ""))

                # Fields 4-6: Grouping of Items checkboxes
                LOGGER.info("  [Main] Grouping of Items checkboxes...")
                await _set_checkbox_tab(
                    "Single Payment for Marked Item",
                    grp.get("Single Payment for Marked Item"),
                    webgui_frame,
                )
                await _set_checkbox_tab(
                    "Payment per Due Day", grp.get("Payment per Due Day"), webgui_frame
                )
                await _set_checkbox_tab(
                    "Extended Individual Payment",
                    grp.get("Extended Individual Payment"),
                    webgui_frame,
                )

                # Fields 7-9: Foreign Payments checkboxes
                LOGGER.info("  [Main] Foreign Payments checkboxes...")
                await _set_checkbox_tab(
                    "Foreign Business Partner Allowed",
                    fgn.get("Foreign Business Partner Allowed"),
                    webgui_frame,
                )
                await _set_checkbox_tab(
                    "Foreign Currency Allowed",
                    fgn.get("Foreign Currency Allowed"),
                    webgui_frame,
                )
                await _set_checkbox_tab(
                    "Bank Abroad Allowed", fgn.get("Bank Abroad Allowed"), webgui_frame
                )

                # Fields 10-12: Bank Selection Control (radio group)
                # In SAP, Tab moves to the radio group, then Arrow keys change selection
                LOGGER.info("  [Main] Bank Selection Control radio...")
                await page.keyboard.press("Tab")
                await asyncio.sleep(0.5)
                # The radio group is now focused. We need to determine which is selected.
                # For "No Optimization" (1st), don't press arrow. For 2nd, press Down once. For 3rd, Down twice.
                if bsc.get("Optimize by Bank Group"):
                    await page.keyboard.press("ArrowDown")
                    await asyncio.sleep(0.3)
                    LOGGER.info("    ✔ Radio → 'Optimize by Bank Group'")
                elif bsc.get("Optimize by Postal Code"):
                    await page.keyboard.press("ArrowDown")
                    await asyncio.sleep(0.3)
                    await page.keyboard.press("ArrowDown")
                    await asyncio.sleep(0.3)
                    LOGGER.info("    ✔ Radio → 'Optimize by Postal Code'")
                else:
                    LOGGER.info("    ✔ Radio → 'No Optimization' (default/first)")

                # ═══════════════════════════════════════════════
                #  FORM DATA — Verification, then Tab
                # ═══════════════════════════════════════════════
                forms_data = data.get("Forms", {})
                sorting_data = data.get("Sorting of the", {})
                drawer_data = data.get("Drawer on the form", {})

                if forms_data or sorting_data or drawer_data:
                    LOGGER.info("  [Form Data] Clicking button...")
                    await _click_sap_button(webgui_frame, "Form Data")

                    # Dump the labels visible AFTER expansion to verify
                    post_expand_labels = await webgui_frame.evaluate("""() => {
                        const labels = [];
                        document.querySelectorAll('span, label, td').forEach(el => {
                            const t = el.textContent.trim();
                            if (t.length > 2 && t.length < 50 && !t.includes('{'))
                                labels.push(t.substring(0, 50));
                        });
                        return [...new Set(labels)].slice(0, 30);
                    }""")
                    LOGGER.info(
                        f"  Labels after Form Data expand: {post_expand_labels}"
                    )

                    # -- Handle the dropdown (PDF / SAPScript) --
                    medium_type = forms_data.get("Payment Medium Type", "")
                    form_name = forms_data.get("Form for the Payment Medium", "")

                    if medium_type or form_name:
                        LOGGER.info("  [Form Data] Filling Payment Medium...")
                        # Find the dropdown near "Payment Medium" label
                        if medium_type:
                            dd_set = await webgui_frame.evaluate(
                                """(targetVal) => {
                                // Strategy 1: Native <select> near "Payment Medium"
                                const allLabels = document.querySelectorAll('span, label, td');
                                for (const lbl of allLabels) {
                                    const t = lbl.textContent.trim();
                                    if (t.includes('Payment Medium') && t.length < 80) {
                                        let parent = lbl.parentElement;
                                        for (let i = 0; i < 10; i++) {
                                            if (!parent) break;
                                            // Try <select>
                                            const sel = parent.querySelector('select');
                                            if (sel) {
                                                for (const opt of sel.options) {
                                                    if (opt.text.toLowerCase().includes(targetVal.toLowerCase()) ||
                                                        opt.value.toLowerCase().includes(targetVal.toLowerCase())) {
                                                        sel.value = opt.value;
                                                        sel.dispatchEvent(new Event('change', {bubbles: true}));
                                                        return 'select:' + opt.text;
                                                    }
                                                }
                                            }
                                            // Try SAP lsComboBox (<input> with aria-role or custom class)
                                            const combo = parent.querySelector(
                                                '[class*="lsComboBox"] input, ' +
                                                '[class*="Combo"] input, ' +
                                                'input[role="combobox"], ' +
                                                'input[aria-haspopup]'
                                            );
                                            if (combo) {
                                                combo.scrollIntoView({behavior: 'instant', block: 'center'});
                                                combo.focus();
                                                combo.click();
                                                combo.value = targetVal;
                                                combo.dispatchEvent(new Event('input', {bubbles: true}));
                                                combo.dispatchEvent(new Event('change', {bubbles: true}));
                                                return 'combobox:' + targetVal;
                                            }
                                            parent = parent.parentElement;
                                        }
                                    }
                                }
                                // Fallback: find any <select> on page
                                const allSelects = document.querySelectorAll('select');
                                for (const sel of allSelects) {
                                    for (const opt of sel.options) {
                                        if (opt.text.toLowerCase().includes(targetVal.toLowerCase())) {
                                            sel.value = opt.value;
                                            sel.dispatchEvent(new Event('change', {bubbles: true}));
                                            return 'fallback-select:' + opt.text;
                                        }
                                    }
                                }
                                return null;
                            }""",
                                medium_type,
                            )
                            if dd_set:
                                LOGGER.info(f"    ✔ Dropdown set: {dd_set}")
                            else:
                                # Last-resort: Playwright approach
                                LOGGER.warning(
                                    f"    JS dropdown failed. Trying Playwright select..."
                                )
                                try:
                                    # Try to find a select near "Payment Medium" text
                                    sel_loc = webgui_frame.locator("select").first
                                    options = await sel_loc.evaluate("""el => {
                                        return Array.from(el.options).map(o => ({value: o.value, text: o.text}));
                                    }""")
                                    LOGGER.info(f"    Select options found: {options}")
                                    # Try to select by label
                                    await sel_loc.select_option(label=medium_type)
                                    LOGGER.info(
                                        f"    ✔ Dropdown set via Playwright select_option"
                                    )
                                except Exception as sel_e:
                                    LOGGER.warning(
                                        f"    Playwright select also failed: {sel_e}"
                                    )
                                    # Absolute last resort: type into the first input-like element
                                    try:
                                        combo = webgui_frame.locator(
                                            "[class*='Combo'], [role='combobox']"
                                        ).first
                                        await combo.click(timeout=3000)
                                        await asyncio.sleep(0.3)
                                        await page.keyboard.press("Control+a")
                                        await page.keyboard.type(medium_type)
                                        await page.keyboard.press("Enter")
                                        LOGGER.info(
                                            f"    ✔ Dropdown set via type+enter"
                                        )
                                    except Exception as combo_e:
                                        LOGGER.error(
                                            f"    All dropdown strategies failed: {combo_e}"
                                        )
                            await asyncio.sleep(1)

                        # Fill form name text field
                        if form_name:
                            found = await webgui_frame.evaluate("""() => {
                                const allEls = document.querySelectorAll('span, label, td');
                                for (const el of allEls) {
                                    const t = el.textContent.trim();
                                    if (t.includes('Payment Medium') && t.length < 80) {
                                        let parent = el.parentElement;
                                        for (let i = 0; i < 8; i++) {
                                            if (!parent) break;
                                            const inputs = parent.querySelectorAll('input[type="text"], input:not([type])');
                                            for (const inp of inputs) {
                                                if (!inp.disabled && !inp.readOnly) {
                                                    inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                                    inp.focus();
                                                    inp.click();
                                                    return true;
                                                }
                                            }
                                            parent = parent.parentElement;
                                        }
                                    }
                                }
                                return false;
                            }""")
                            if found:
                                await asyncio.sleep(0.3)
                                await _type_field(form_name)
                                LOGGER.info(
                                    f"    ✔ 'Form for the Payment Medium' = '{form_name}'"
                                )
                            else:
                                LOGGER.warning(
                                    f"    ✘ Could not find form name text input"
                                )

                    # Drawer fields: Tab through 4 sequential text inputs
                    d1 = drawer_data.get("label_text1", "")
                    d2 = drawer_data.get("label_text2", "")
                    d3 = drawer_data.get("label_text3", "")
                    d4 = drawer_data.get("label_text4", "")

                    if d1 or d2 or d3 or d4:
                        LOGGER.info("  [Form Data] Drawer fields...")
                        # Find and click the first Drawer input to anchor
                        drawer_clicked = await webgui_frame.evaluate("""() => {
                            const allEls = document.querySelectorAll('span, label, td');
                            for (const el of allEls) {
                                const t = el.textContent.trim();
                                if (t.includes('Drawer') && t.length < 30) {
                                    let parent = el.parentElement;
                                    for (let i = 0; i < 6; i++) {
                                        if (!parent) break;
                                        const inp = parent.querySelector('input[type="text"], input:not([type])');
                                        if (inp && !inp.disabled && !inp.readOnly) {
                                            inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                            inp.focus();
                                            inp.click();
                                            return true;
                                        }
                                        parent = parent.parentElement;
                                    }
                                }
                            }
                            return false;
                        }""")
                        if drawer_clicked:
                            # Already on Drawer 1
                            if d1:
                                await _type_field(d1)
                                LOGGER.info(f"    ✔ 'Drawer Line 1' = '{d1}'")
                            await _tab_and_type("Drawer Line 2", d2)
                            await _tab_and_type("Drawer Line 3", d3)
                            await _tab_and_type("Drawer Line 4", d4)
                        else:
                            LOGGER.warning(
                                "    Drawer field not found, trying Tab sequence..."
                            )
                            await _tab_and_type("Drawer Line 1", d1)
                            await _tab_and_type("Drawer Line 2", d2)
                            await _tab_and_type("Drawer Line 3", d3)
                            await _tab_and_type("Drawer Line 4", d4)

                    # Sorting fields
                    LOGGER.info("  [Form Data] Sorting fields...")
                    corr = sorting_data.get("Correspondence", "")
                    line_items = sorting_data.get("Line Items", "")
                    if corr:
                        # Find Correspondence input by label
                        corr_found = await webgui_frame.evaluate("""() => {
                            const allEls = document.querySelectorAll('span, label, td');
                            for (const el of allEls) {
                                const t = el.textContent.trim();
                                if (t.includes('Correspondence') && t.length < 30) {
                                    let parent = el.parentElement;
                                    for (let i = 0; i < 6; i++) {
                                        if (!parent) break;
                                        const inp = parent.querySelector('input[type="text"], input:not([type])');
                                        if (inp && !inp.disabled && !inp.readOnly) {
                                            inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                            inp.focus();
                                            inp.click();
                                            return true;
                                        }
                                        parent = parent.parentElement;
                                    }
                                }
                            }
                            return false;
                        }""")
                        if corr_found:
                            await asyncio.sleep(0.3)
                            await _type_field(corr)
                            LOGGER.info(f"    ✔ 'Correspondence' = '{corr}'")
                        else:
                            await _tab_and_type("Correspondence", corr)

                    if line_items:
                        await _tab_and_type("Line Items", line_items)

                # ═══════════════════════════════════════════════
                #  PYT ADV. CONTROL — Scroll down, click, verify
                # ═══════════════════════════════════════════════
                adv_data = data.get("Payment Advice Note Control", {})
                if adv_data:
                    LOGGER.info("  [Pyt Adv. Control] Scrolling page down...")
                    # Scroll down within the webgui frame to reveal the section
                    await webgui_frame.evaluate("""() => {
                        const container = document.querySelector('[class*="lsScrollContainer"], [class*="scroll"], .sapMScrollCont')
                            || document.documentElement;
                        container.scrollTop = container.scrollHeight;
                    }""")
                    await asyncio.sleep(1)
                    # Also try scrolling via Page Down key
                    await page.keyboard.press("PageDown")
                    await asyncio.sleep(1)

                    LOGGER.info("  [Pyt Adv. Control] Clicking button...")
                    await _click_sap_button(webgui_frame, "Pyt Adv. Control")

                    # Dump ALL labels (60) to see what's visible after expansion
                    adv_labels = await webgui_frame.evaluate("""() => {
                        const labels = [];
                        document.querySelectorAll('span, label, td').forEach(el => {
                            const t = el.textContent.trim();
                            if (t.length > 2 && t.length < 60 && !t.includes('{'))
                                labels.push(t.substring(0, 60));
                        });
                        return [...new Set(labels)].slice(0, 60);
                    }""")
                    LOGGER.info(f"  Labels after Pyt Adv expand: {adv_labels}")

                    # ── Helper: click a radio button by its label text ──
                    async def _click_radio(wf, label_text):
                        """Click an SAP radio button by finding the label, then clicking
                        the nearby input[type=radio] or the label itself via Playwright."""
                        # Strategy 1: JS — find label, walk to radio input
                        clicked = await wf.evaluate(
                            """(labelText) => {
                            const allEls = document.querySelectorAll('span, label, td');
                            for (const el of allEls) {
                                const t = el.textContent.trim();
                                if (t === labelText ||
                                    (t.toLowerCase() === labelText.toLowerCase() && t.length < labelText.length + 5)) {
                                    // Walk up to find a radio input
                                    let parent = el.parentElement;
                                    for (let i = 0; i < 6; i++) {
                                        if (!parent) break;
                                        const radio = parent.querySelector('input[type="radio"]');
                                        if (radio) {
                                            radio.scrollIntoView({behavior: 'instant', block: 'center'});
                                            radio.focus();
                                            radio.click();
                                            radio.checked = true;
                                            radio.dispatchEvent(new Event('change', {bubbles: true}));
                                            return 'radio-input';
                                        }
                                        parent = parent.parentElement;
                                    }
                                    // No radio found — click the label itself
                                    el.scrollIntoView({behavior: 'instant', block: 'center'});
                                    el.click();
                                    return 'label-click';
                                }
                            }
                            return null;
                        }""",
                            label_text,
                        )

                        if clicked:
                            LOGGER.info(
                                f"    ✔ Radio '{label_text}' clicked via {clicked}"
                            )
                        else:
                            # Fallback: Playwright get_by_text
                            try:
                                el = wf.get_by_text(label_text, exact=True).first
                                await el.click(timeout=3000)
                                LOGGER.info(
                                    f"    ✔ Radio '{label_text}' clicked via Playwright"
                                )
                            except Exception as e:
                                LOGGER.warning(
                                    f"    ✘ Radio '{label_text}' click failed: {e}"
                                )
                        await asyncio.sleep(0.5)

                    # ── Helper: fill a text input near a label ──
                    async def _fill_near_label(wf, label_text, value):
                        """Find and fill a text input near a label."""
                        found = await wf.evaluate(
                            """(labelText) => {
                            const allEls = document.querySelectorAll('span, label, td');
                            for (const el of allEls) {
                                const t = el.textContent.trim();
                                if (t === labelText || t.toLowerCase() === labelText.toLowerCase()) {
                                    let parent = el.parentElement;
                                    for (let i = 0; i < 6; i++) {
                                        if (!parent) break;
                                        const inp = parent.querySelector('input[type="text"], input:not([type])');
                                        if (inp && !inp.disabled && !inp.readOnly) {
                                            inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                            inp.focus();
                                            inp.click();
                                            return true;
                                        }
                                        parent = parent.parentElement;
                                    }
                                }
                            }
                            return false;
                        }""",
                            label_text,
                        )
                        if found:
                            await asyncio.sleep(0.3)
                            await _type_field(value)
                            LOGGER.info(f"    ✔ '{label_text}' = '{value}'")
                        else:
                            LOGGER.warning(
                                f"    ✘ Could not find input for '{label_text}'"
                            )

                    # ── Process the Pyt Adv. Control fields ──
                    # Radio group 1: "restricted to" / "None" / "as many as req"
                    radio_group_1 = ["restricted to", "None", "as many as req"]
                    selected_radio = None
                    for r in radio_group_1:
                        if adv_data.get(r) is True:
                            selected_radio = r
                            break  # First True radio wins (they're mutually exclusive)

                    if selected_radio:
                        LOGGER.info(f"  [Pyt Adv] Selecting radio: '{selected_radio}'")
                        await _click_radio(webgui_frame, selected_radio)

                    # "Rows" text field (next to "restricted to")
                    rows_val = adv_data.get("rows", "")
                    if rows_val:
                        LOGGER.info(f"  [Pyt Adv] Filling Rows...")
                        await _fill_near_label(webgui_frame, "Rows", rows_val)

                    # "Do Not Consider Item Text" — this might be a checkbox
                    dncit = adv_data.get("Do Not Consider Item Text", None)
                    if dncit is not None:
                        LOGGER.info(
                            f"  [Pyt Adv] Handling 'Do Not Consider Item Text'..."
                        )
                        await _set_checkbox_tab(
                            "Do Not Consider Item Text", dncit, webgui_frame
                        )

                    # ── Radio group 2: "Payment advice output according to no. of lines" ──
                    adv_output_data = data.get(
                        "Payment advice output according to no. of lines", {}
                    )
                    if adv_output_data:
                        LOGGER.info(
                            "  [Pyt Adv] Processing 'Payment advice output according to no. of lines' section..."
                        )

                        # Radio Group 2
                        radio_group_2 = [
                            "Pymt adv. after ... lines",
                            "Always Payt Adv",
                            "NoPytAdv",
                        ]
                        selected_radio_2 = None
                        for r in radio_group_2:
                            if adv_output_data.get(r) is True:
                                selected_radio_2 = r
                                break

                        if selected_radio_2:
                            LOGGER.info(f"    Selected radio: '{selected_radio_2}'")
                            await _click_radio(webgui_frame, selected_radio_2)

                        # Sub-options (Checkboxes/Radios)
                        sub_options = [
                            "Pymt mthd valid to ... lines",
                            "Distribute items, .. lines per pymt",
                        ]
                        for opt in sub_options:
                            val = adv_output_data.get(opt)
                            if val is not None:
                                LOGGER.info(
                                    f"    Handling sub-option: '{opt}' -> {val}"
                                )
                                # Use JS to click since they might be radios or checkboxes
                                await webgui_frame.evaluate(
                                    """(args) => {
                                    const {labelText, targetVal} = args;
                                    const allLabels = document.querySelectorAll('span, label, td');
                                    for (const el of allLabels) {
                                        const t = el.textContent.trim();
                                        if (t.includes(labelText)) {
                                            let parent = el.parentElement;
                                            for (let i = 0; i < 6; i++) {
                                                if (!parent) break;
                                                const inp = parent.querySelector('input');
                                                if (inp) {
                                                    if (inp.checked !== targetVal) {
                                                        inp.click();
                                                    }
                                                    return true;
                                                }
                                                parent = parent.parentElement;
                                            }
                                        }
                                    }
                                    return false;
                                }""",
                                    {"labelText": opt, "targetVal": val},
                                )
                                await asyncio.sleep(0.5)

                # ═══════════════════════════════════════════════
                #  SAVE
                # ═══════════════════════════════════════════════
                LOGGER.info("  Saving changes...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page)

                # Back to list
                await page.keyboard.press("F3")
                await asyncio.sleep(2)
                LOGGER.info(f"  ✔ CoCd {cocd} / Payt {pyt_meth} complete.")

            LOGGER.info("DONE.")
        except Exception as e:
            LOGGER.error(f"Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            await browser.close()



async def Set_Up_Payment_Methods_for_Each_Country_Region_for_Payment_Transactions_101972(
    targets: list[dict],
):
    """
    SSCUI 101972 – Set Up Payment Methods for Each Country/Region for Payment Transactions.
    ─────────────────────────────────────────────────────────────────────────────────────────
    Automates the \"Payment Method/Country\" configuration screen (VC_T042ZL).

    Main list view columns:
        C/R | Country/Reg. Name | Payt Meth. | Name

    Dialog Structure (left panel):
        ├── Payment Method/Country  (main view)
        ├── Currencies Allowed
        ├── Permitted Destination Countries
        ├── Note to Payee by Origin
        └── Permitted Payment System for

    Detail view fields (after double-click or New Entries):
        ── Keys ──
        Country/Reg. (text)          – e.g. "IN"
        Pymt Meth. (text)            – e.g. "T"

        ── Description ──
        Description (text)           – e.g. "Bank Transfer"

        ── Payment method for ── (radio group)
        Outgoing payments / Incoming payments

        ── Payment method classification ── (radio group)
        Bank transf / Check / Bill/ex / Check/bill/ex. / Supplier Fin.

        ── Checkboxes ──
        Post office curr.acct method?   |  Bill of exch. accepted
        Allowed for personnel payments  |  ISR Payment Procedure
        Create bill/exch.before due date|  EU Internal Transfer

        ── Required master record specifications ── (checkboxes)
        Street,P.O.box or P.O.box pst code
        Bank details
        Account Number Required
        IBAN Required
        SWIFT Code Required
        Alias Required
        Collection authorization
        SEPA Mandate Required

        ── Posting Details ──
        Document Type for Payment (text) – e.g. "ZP"
        Clearing Document Type (text)    – e.g. "ZV"
        Sp.G/L Ind.B/Ex. / B/Ex.Pmnt Req. (text)
        Sp.G/L Ind. for SF (text)
        Payment Order Only (checkbox)

        ── Payment medium ──
        Format (text)              – e.g. "IN_CGI_XML_CT"
        Format supplement (text)

    Target format (supports both string and dict for radio groups):
        {
            "Country_Region": "IN",
            "Pymt_Meth": "T",
            "Description": "Bank Transfer",
            "Description_2": "Bank Transfer",  # optional; defaults to Description if omitted
            "Payment_Method_For": {"Outgoing payments": True, "Incoming payments": False},
            "Payment_Method_Classification": {"Bank transf": True, "Check": False, "Bill/ex": False,
                                                "Check/bill/ex.": False, "Supplier Fin.": False},
            "Post_Office_Curr_Acct_Method": False,
            "Allowed_For_Personnel_Payments": True,
            "Create_Bill_Exch_Before_Due_Date": False,
            "Bill_Of_Exch_Accepted": False,
            "ISR_Payment_Procedure": False,
            "EU_Internal_Transfer": False,
            "Street_PO_Box": False,
            "Bank_Details": True,
            "Account_Number_Required": False,
            "IBAN_Required": True,
            "SWIFT_Code_Required": False,
            "Alias_Required": False,
            "Collection_Authorization": False,
            "SEPA_Mandate_Required": False,
            "Document_Type_For_Payment": "ZP",
            "Clearing_Document_Type": "ZV",
            "SpGL_Ind_BEx": "",
            "SpGL_Ind_SF": "",
            "Payment_Order_Only": False,
            "Format": "IN_CGI_XML_CT",
            "Format_Supplement": ""
        }
    """
    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=FIAPPY_V_T042ZL&CustomizingObject=VC_T042ZL&CustomizingObjectType=C&CustomizingProject=&CustomizingTransaction=S_ALR_87100689&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        # ── Local helpers ──

        async def _type_field(value):
            """Select all → delete → type new value."""
            await page.keyboard.press("Control+a")
            await asyncio.sleep(0.2)
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.2)
            await page.keyboard.type(str(value), delay=50)
            await asyncio.sleep(0.5)

        async def _tab_and_type(field_name, value):
            """Tab to next field and type a value. If value is empty/None, just Tab past it."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.5)
            if value:
                await _type_field(value)
                LOGGER.info(f"    ✔ '{field_name}' = '{value}'")
            else:
                LOGGER.info(f"    ⏭ '{field_name}' (skipped)")

        async def _click_first_input(wf, label_text):
            """Click the first input field near a label to anchor the Tab sequence."""
            found = await wf.evaluate(
                """(labelText) => {
                const allElements = Array.from(document.querySelectorAll('span, label, td'));
                for (const el of allElements) {
                    const t = el.textContent.trim();
                    if (t.includes(labelText) && t.length < labelText.length + 15) {
                        // Strategy 1: Walk up the parent chain looking for an input
                        let parent = el.parentElement;
                        for (let i = 0; i < 8; i++) {
                            if (!parent) break;
                            const inp = parent.querySelector('input[type="text"], input:not([type]), input[type="number"]');
                            if (inp && !inp.disabled && !inp.readOnly) {
                                inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                inp.focus();
                                inp.click();
                                return true;
                            }
                            parent = parent.parentElement;
                        }

                        // Strategy 2: Check next sibling elements (SAP sometimes puts label and input side-by-side)
                        let sibling = el.nextElementSibling;
                        for (let i = 0; i < 5 && sibling; i++) {
                            const inp = sibling.querySelector ? sibling.querySelector('input[type="text"], input:not([type])') : null;
                            if (inp && !inp.disabled && !inp.readOnly) {
                                inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                inp.focus();
                                inp.click();
                                return true;
                            }
                            if (sibling.tagName === 'INPUT' && !sibling.disabled && !sibling.readOnly) {
                                sibling.scrollIntoView({behavior: 'instant', block: 'center'});
                                sibling.focus();
                                sibling.click();
                                return true;
                            }
                            sibling = sibling.nextElementSibling;
                        }

                        // Strategy 3: Check parent's next sibling (table row pattern: label in one cell, input in next)
                        let parentTd = el.closest('td');
                        if (parentTd && parentTd.nextElementSibling) {
                            const inp = parentTd.nextElementSibling.querySelector('input[type="text"], input:not([type])');
                            if (inp && !inp.disabled && !inp.readOnly) {
                                inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                inp.focus();
                                inp.click();
                                return true;
                            }
                        }
                    }
                }
                return false;
            }""",
                label_text,
            )
            return found

        async def _set_checkbox_by_label(wf, label_text, target_state):
            """Find a checkbox near a label and set it to the target state."""
            if target_state is None:
                LOGGER.info(f"    ⏭ Checkbox '{label_text}': skipped (None)")
                return

            result = await wf.evaluate(
                """(args) => {
                const {labelText, targetState} = args;
                const allEls = document.querySelectorAll('span, label, td');
                for (const el of allEls) {
                    const t = el.textContent.trim();
                    if (t.includes(labelText) && t.length < labelText.length + 20) {
                        // Walk up to find a checkbox input
                        let parent = el.parentElement;
                        for (let i = 0; i < 8; i++) {
                            if (!parent) break;
                            const inp = parent.querySelector('input[type="checkbox"]');
                            if (inp) {
                                const isChecked = inp.checked;
                                if (isChecked !== targetState) {
                                    inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                    inp.click();
                                    return {toggled: true, was: isChecked, now: targetState};
                                }
                                return {toggled: false, was: isChecked, now: targetState};
                            }
                            // Check lsdata-based SAP checkbox
                            const sapChk = parent.querySelector('[lsdata]');
                            if (sapChk) {
                                try {
                                    const raw = sapChk.getAttribute('lsdata');
                                    const d = JSON.parse(raw.replace(/'/g, '"'));
                                    const isChecked = d['1'] === true || d['1'] === '1' || d['1'] === 1;
                                    if (isChecked !== targetState) {
                                        sapChk.scrollIntoView({behavior: 'instant', block: 'center'});
                                        sapChk.click();
                                        return {toggled: true, was: isChecked, now: targetState};
                                    }
                                    return {toggled: false, was: isChecked, now: targetState};
                                } catch(e) {}
                            }
                            parent = parent.parentElement;
                        }
                    }
                }
                return null;
            }""",
                {"labelText": label_text, "targetState": target_state},
            )

            if result:
                if result.get("toggled"):
                    LOGGER.info(
                        f"    ✔ Checkbox '{label_text}': toggled {result['was']} → {result['now']}"
                    )
                else:
                    LOGGER.info(
                        f"    ✔ Checkbox '{label_text}': already correct ({result['was']})"
                    )
                await asyncio.sleep(0.5)
            else:
                # Fallback: Playwright locator approach
                LOGGER.warning(
                    f"    JS checkbox failed for '{label_text}'. Trying Playwright..."
                )
                try:
                    chk = wf.get_by_text(label_text, exact=False).first
                    await chk.scroll_into_view_if_needed(timeout=3000)
                    # Find closest checkbox input
                    parent = chk.locator(
                        "xpath=ancestor::tr[1]//input[@type='checkbox']"
                    ).first
                    if await parent.count() > 0:
                        await parent.click(force=True, timeout=3000)
                        LOGGER.info(
                            f"    ✔ Checkbox '{label_text}': toggled via Playwright"
                        )
                    else:
                        await chk.click(force=True, timeout=3000)
                        LOGGER.info(
                            f"    ✔ Checkbox '{label_text}': clicked label via Playwright"
                        )
                except Exception as e:
                    LOGGER.warning(
                        f"    ✘ Could not toggle checkbox '{label_text}': {e}"
                    )
                await asyncio.sleep(0.5)

        async def _click_radio(wf, label_text):
            """Click an SAP radio button by finding the label, then clicking the nearby radio input."""
            clicked = await wf.evaluate(
                """(labelText) => {
                const allEls = document.querySelectorAll('span, label, td');
                for (const el of allEls) {
                    const t = el.textContent.trim();
                    if (t === labelText ||
                        (t.toLowerCase() === labelText.toLowerCase() && t.length < labelText.length + 5)) {
                        // Walk up to find a radio input
                        let parent = el.parentElement;
                        for (let i = 0; i < 6; i++) {
                            if (!parent) break;
                            const radio = parent.querySelector('input[type="radio"]');
                            if (radio) {
                                radio.scrollIntoView({behavior: 'instant', block: 'center'});
                                radio.focus();
                                radio.click();
                                radio.checked = true;
                                radio.dispatchEvent(new Event('change', {bubbles: true}));
                                return 'radio-input';
                            }
                            parent = parent.parentElement;
                        }
                        // No radio found — click the label itself
                        el.scrollIntoView({behavior: 'instant', block: 'center'});
                        el.click();
                        return 'label-click';
                    }
                }
                return null;
            }""",
                label_text,
            )

            if clicked:
                LOGGER.info(f"    ✔ Radio '{label_text}' clicked via {clicked}")
            else:
                # Fallback: Playwright get_by_text
                try:
                    el = wf.get_by_text(label_text, exact=True).first
                    await el.click(timeout=3000)
                    LOGGER.info(f"    ✔ Radio '{label_text}' clicked via Playwright")
                except Exception as e:
                    LOGGER.warning(f"    ✘ Radio '{label_text}' click failed: {e}")
            await asyncio.sleep(0.5)

        async def _click_sap_button(wf, button_text):
            """Click an SAP WebGUI toolbar button by text."""
            success = False
            try:
                btn = wf.get_by_text(button_text, exact=False).first
                await btn.scroll_into_view_if_needed(timeout=3000)
                await btn.click(timeout=5000)
                LOGGER.info(f"    Button '{button_text}' clicked via Playwright")
                success = True
            except Exception as e1:
                LOGGER.warning(f"    Playwright click failed for '{button_text}': {e1}")
                try:
                    escaped = button_text.replace(" ", ".*")
                    btn = wf.locator(f"text=/{escaped}/i").first
                    await btn.scroll_into_view_if_needed(timeout=3000)
                    await btn.click(timeout=5000)
                    LOGGER.info(f"    Button '{button_text}' clicked via regex")
                    success = True
                except Exception as e2:
                    LOGGER.warning(f"    Regex click also failed: {e2}")
                    # JS fallback
                    try:
                        await wf.evaluate(
                            """(btnText) => {
                            const allEls = document.querySelectorAll('span, div, a, td, button, [role="button"]');
                            for (const el of allEls) {
                                const t = (el.textContent || '').trim();
                                if (t.toLowerCase().includes(btnText.toLowerCase()) && t.length < btnText.length + 15) {
                                    el.scrollIntoView({behavior: 'instant', block: 'center'});
                                    el.click();
                                    return true;
                                }
                            }
                            return false;
                        }""",
                            button_text,
                        )
                        LOGGER.info(
                            f"    Button '{button_text}' clicked via JS fallback"
                        )
                        success = True
                    except Exception as e3:
                        LOGGER.error(
                            f"    All click strategies failed for '{button_text}': {e3}"
                        )
            if success:
                await asyncio.sleep(4)
            return success

        async def _find_labeled_inputs(wf, label_text):
            """
            Return ranked text-like inputs associated with a label text.
            Ranking prefers direct title/aria matches, then nearest input to label.
            This is used for screens that contain duplicate labels (e.g., two Description fields).
            """
            try:
                return (
                    await wf.evaluate(
                        """(labelText) => {
                    const normalize = (s) => (s || '')
                        .replace(/\\s+/g, ' ')
                        .replace(/:+$/g, '')
                        .trim()
                        .toLowerCase();
                    const needle = normalize(labelText);
                    const seen = new Map();

                    const isTextInput = (inp) => {
                        if (!inp || !inp.id || inp.disabled) return false;
                        const type = (inp.getAttribute('type') || 'text').toLowerCase();
                        if (['checkbox', 'radio', 'hidden', 'submit', 'button'].includes(type)) return false;
                        if (inp.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) return false;
                        return true;
                    };

                    const pushInput = (inp, score, sourceText, sourceType) => {
                        if (!isTextInput(inp)) return;
                        const r = inp.getBoundingClientRect();
                        const candidate = {
                            id: inp.id,
                            readOnly: !!inp.readOnly,
                            disabled: !!inp.disabled,
                            title: inp.getAttribute('title') || '',
                            aria: inp.getAttribute('aria-label') || '',
                            label: sourceText || '',
                            source: sourceType || '',
                            score: Number(score || 0),
                            top: r.top,
                            left: r.left
                        };
                        const prev = seen.get(inp.id);
                        if (!prev || candidate.score > prev.score) {
                            seen.set(inp.id, candidate);
                        }
                    };

                    const scoreFromDistance = (lblEl, inp, baseScore) => {
                        const lr = lblEl.getBoundingClientRect();
                        const ir = inp.getBoundingClientRect();
                        const dy = Math.abs((ir.top + ir.height / 2) - (lr.top + lr.height / 2));
                        const dx = ir.left - lr.right;
                        // Penalize fields that are left of label, favor same row / right-side cells
                        const dxPenalty = dx < -8 ? 80 + Math.abs(dx) : Math.abs(dx) * 0.6;
                        const penalty = dy * 2 + dxPenalty;
                        return baseScore - Math.min(60, penalty / 4);
                    };

                    // Pass 1: match by input title / aria-label (very reliable on SAP controls)
                    const allInputs = Array.from(document.querySelectorAll('input, textarea'));
                    for (const inp of allInputs) {
                        const title = normalize(inp.getAttribute('title'));
                        const aria = normalize(inp.getAttribute('aria-label'));
                        let score = 0;
                        if ((title && title === needle) || (aria && aria === needle)) {
                            score = 140;
                        } else if ((title && title.startsWith(needle)) || (aria && aria.startsWith(needle))) {
                            score = 120;
                        } else if ((title && title.includes(needle)) || (aria && aria.includes(needle))) {
                            score = 100;
                        }
                        if (score > 0) {
                            pushInput(inp, score, title || aria, "title-aria");
                        }
                    }

                    // Pass 2: match by nearby label text
                    const labels = Array.from(document.querySelectorAll('label, span, td, div'));
                    for (const el of labels) {
                        const txt = normalize(el.textContent);
                        if (!txt) continue;
                        const exact = txt === needle;
                        const starts = txt.startsWith(needle);
                        const includes = txt.includes(needle);
                        if (!(exact || starts || includes)) continue;
                        if (el.closest('[class*="urTbar"], [class*="lsToolbar"], [role="toolbar"]')) continue;
                        if (txt.length > needle.length + 40) continue;

                        const base = exact ? 90 : (starts ? 78 : 65);

                        // Strong direct association via <label for="">
                        if (el.tagName === 'LABEL' && el.htmlFor) {
                            const direct = document.getElementById(el.htmlFor);
                            if (direct) {
                                pushInput(direct, base + 25, txt, "label-for");
                            }
                        }

                        // Nearby candidates in row/container ordered by geometric proximity
                        const containers = [
                            el.closest('tr'),
                            el.parentElement,
                            el.closest('table')
                        ];
                        for (const container of containers) {
                            if (!container) continue;
                            for (const inp of container.querySelectorAll('input, textarea')) {
                                const score = scoreFromDistance(el, inp, base);
                                pushInput(inp, score, txt, "container-near");
                            }
                        }

                        // Next siblings (common SAP label/value cell pattern)
                        let sib = el.nextElementSibling;
                        for (let i = 0; i < 4 && sib; i++) {
                            if (sib.tagName === 'INPUT' || sib.tagName === 'TEXTAREA') {
                                const score = scoreFromDistance(el, sib, base + 10);
                                pushInput(sib, score, txt, "sibling");
                            }
                            if (sib.querySelectorAll) {
                                for (const inp of sib.querySelectorAll('input, textarea')) {
                                    const score = scoreFromDistance(el, inp, base + 8);
                                    pushInput(inp, score, txt, "sibling-child");
                                }
                            }
                            sib = sib.nextElementSibling;
                        }
                    }

                    const results = Array.from(seen.values());
                    results.sort((a, b) => {
                        if (b.score !== a.score) return b.score - a.score;
                        if (a.top !== b.top) return a.top - b.top;
                        return a.left - b.left;
                    });
                    return results;
                }""",
                        label_text,
                    )
                    or []
                )
            except Exception as e:
                LOGGER.warning(
                    f"    Failed to find inputs for label '{label_text}': {e}"
                )
                return []

        async def _choose_labeled_input_id(
            wf, label_text, occurrence=0, exclude_ids=None
        ):
            """Pick one ranked input id for a label, optionally excluding known ids."""
            exclude = {x for x in (exclude_ids or []) if x}
            candidates = await _find_labeled_inputs(wf, label_text)
            filtered = [c for c in candidates if c.get("id") not in exclude]

            if not filtered:
                LOGGER.warning(
                    f"    ✘ No candidate input left for '{label_text}' "
                    f"(total={len(candidates)}, excluded={len(exclude)})"
                )
                return None
            if occurrence >= len(filtered):
                LOGGER.warning(
                    f"    ✘ Label '{label_text}' occurrence {occurrence + 1} not found "
                    f"(available={len(filtered)})"
                )
                return None

            chosen = filtered[occurrence]
            LOGGER.info(
                f"    ↳ '{label_text}' using id={chosen.get('id')} "
                f"(score={chosen.get('score')}, source={chosen.get('source')})"
            )
            return chosen.get("id")

        async def _set_input_by_id(wf, input_id, value, field_name, commit_key="Tab"):
            """Focus a specific input by DOM id, replace value, and commit with Tab/Enter."""
            try:
                loc = wf.locator(f"id={input_id}")
                if await loc.count() == 0:
                    LOGGER.warning(
                        f"    ✘ Input not found for '{field_name}' (id={input_id})"
                    )
                    return False

                active_id = await wf.evaluate(
                    """(id) => {
                    const el = document.getElementById(id);
                    if (!el) return null;
                    el.scrollIntoView({behavior: 'instant', block: 'center', inline: 'center'});
                    el.focus();
                    el.click();
                    return (document.activeElement && document.activeElement.id) || null;
                }""",
                    input_id,
                )
                await asyncio.sleep(0.2)

                await loc.first.click(force=True, timeout=2500)
                await asyncio.sleep(0.2)

                active_after_click = await wf.evaluate(
                    "() => (document.activeElement && document.activeElement.id) || null"
                )
                if active_after_click != input_id:
                    active_after_click = await wf.evaluate(
                        """(id) => {
                        const el = document.getElementById(id);
                        if (!el) return null;
                        el.dispatchEvent(new MouseEvent('mousedown', {bubbles: true}));
                        el.dispatchEvent(new MouseEvent('mouseup', {bubbles: true}));
                        el.click();
                        el.focus();
                        return (document.activeElement && document.activeElement.id) || null;
                    }""",
                        input_id,
                    )
                    await asyncio.sleep(0.2)

                if active_after_click != input_id:
                    LOGGER.warning(
                        f"    ✘ Focus mismatch for '{field_name}' "
                        f"(target={input_id}, active={active_after_click})"
                    )
                    return False

                # If still read-only, try F2 once (some SAP fields need explicit edit activation)
                read_only = await wf.evaluate(
                    "(id) => !!document.getElementById(id)?.readOnly", input_id
                )
                if read_only:
                    await page.keyboard.press("F2")
                    await asyncio.sleep(0.3)
                    read_only = await wf.evaluate(
                        "(id) => !!document.getElementById(id)?.readOnly", input_id
                    )
                if read_only:
                    LOGGER.warning(f"    ✘ '{field_name}' is read-only (id={input_id})")
                    return False

                await _type_field(value)
                if commit_key:
                    await page.keyboard.press(commit_key)
                    await asyncio.sleep(0.3)
                LOGGER.info(f"    ✔ '{field_name}' = '{value}' (id={input_id})")
                return True
            except Exception as e:
                LOGGER.warning(
                    f"    ✘ Failed to set '{field_name}' (id={input_id}): {e}"
                )
                return False

        async def _set_labeled_input(
            wf,
            label_text,
            value,
            occurrence=0,
            field_name=None,
            commit_key="Tab",
            exclude_ids=None,
        ):
            """
            Set a field by label and ordinal occurrence.
            occurrence=0 targets the first matched field, occurrence=1 the second, etc.
            """
            chosen_id = await _choose_labeled_input_id(
                wf, label_text, occurrence=occurrence, exclude_ids=exclude_ids
            )
            if not chosen_id:
                return None
            name = field_name or f"{label_text}[{occurrence + 1}]"
            success = await _set_input_by_id(
                wf, chosen_id, value, name, commit_key=commit_key
            )
            return chosen_id if success else None

        async def _confirm_save_dialogs():
            """Handle common SAP save confirmation dialogs (Yes/Continue/OK/Save-in-dialog)."""
            confirmed = False
            for _ in range(8):
                for frame in page.frames:
                    try:
                        # 0) Dialog-local "Save" button (avoid toolbar Save confusion)
                        save_in_dialog = frame.locator(
                            ".sapMDialog button:has-text('Save'), "
                            "[role='dialog'] button:has-text('Save'), "
                            ".sapMDialog .lsButton:has-text('Save'), "
                            "[role='dialog'] .lsButton:has-text('Save')"
                        ).first
                        if await save_in_dialog.is_visible(timeout=250):
                            await save_in_dialog.click(force=True)
                            LOGGER.info("  ✔ Clicked 'Save' inside dialog.")
                            confirmed = True
                            await asyncio.sleep(1.5)
                            break

                        # JS fallback for dialog-local Save button (handles non-standard button markup)
                        js_dialog_save = await frame.evaluate("""() => {
                            const dialogs = Array.from(
                                document.querySelectorAll('.sapMDialog, [role="dialog"], .sapUiDlg')
                            ).filter(d => !!(d.offsetParent || d.getClientRects().length));
                            for (const d of dialogs) {
                                const nodes = Array.from(
                                    d.querySelectorAll('button, [role="button"], .sapMBtn, .lsButton, span, div, a')
                                );
                                for (const n of nodes) {
                                    const t = (n.textContent || n.getAttribute('title') || '').trim().toLowerCase();
                                    if (t === 'save') {
                                        n.scrollIntoView({behavior: 'instant', block: 'center'});
                                        n.click();
                                        return true;
                                    }
                                }
                            }
                            return false;
                        }""")
                        if js_dialog_save:
                            LOGGER.info("  ✔ Clicked dialog 'Save' via JS fallback.")
                            confirmed = True
                            await asyncio.sleep(1.5)
                            break

                        yes_btn = frame.locator(
                            "button:has-text('Yes'), [title='Yes'], .sapMBtn:has-text('Yes')"
                        ).first
                        if await yes_btn.is_visible(timeout=300):
                            await yes_btn.click(force=True)
                            LOGGER.info("  ✔ Clicked 'Yes' on save dialog.")
                            confirmed = True
                            await asyncio.sleep(2)
                            break

                        tick = frame.locator(
                            "button[title*='Continue'], "
                            "button:has-text('Continue'), "
                            "button[title*='OK'], "
                            "button:has-text('OK'), "
                            "[title*='Continue (Enter)'], "
                            "[aria-label*='Continue'], "
                            "button[title*='Checkmark'], "
                            ".sapMDialog .sapMBtnEmphasized, "
                            "[role='dialog'] .sapMBtnEmphasized"
                        ).first
                        if await tick.is_visible(timeout=300):
                            await tick.click(force=True)
                            LOGGER.info(
                                "  ✔ Clicked green tick/Continue/OK on save dialog."
                            )
                            confirmed = True
                            await asyncio.sleep(2)
                            break
                    except:
                        pass
                if confirmed:
                    break
                await asyncio.sleep(0.8)
            return confirmed

        async def _click_save_button_anywhere():
            """Click Save button directly (page shell or frame) for SSCUIs where Ctrl+S is ignored."""
            selectors = [
                "button:has-text('Save')",
                "ui5-button:has-text('Save')",
                "[title='Save']",
                "[title*='Save (Ctrl+S)']",
                "[aria-label*='Save']",
                ".sapMBtn:has-text('Save')",
                ".lsButton:has-text('Save')",
                "[role='button']:has-text('Save')",
            ]
            for root in [page, *page.frames]:
                for sel in selectors:
                    try:
                        btn = root.locator(sel).first
                        if await btn.is_visible(timeout=400):
                            await btn.click(force=True)
                            LOGGER.info(f"  ✔ Clicked Save via selector: {sel}")
                            await asyncio.sleep(1.5)
                            return True
                    except:
                        pass
            return False

        async def _click_exit_button_anywhere():
            """Click Exit button directly (page shell or frame), falling back to F3 elsewhere."""
            selectors = [
                "button:has-text('Exit')",
                "ui5-button:has-text('Exit')",
                "[title='Exit']",
                "[title*='Exit']",
                "[aria-label*='Exit']",
                ".sapMBtn:has-text('Exit')",
                ".lsButton:has-text('Exit')",
                "[role='button']:has-text('Exit')",
            ]
            for root in [page, *page.frames]:
                for sel in selectors:
                    try:
                        btn = root.locator(sel).first
                        if await btn.is_visible(timeout=400):
                            await btn.click(force=True)
                            LOGGER.info(f"  ✔ Clicked Exit via selector: {sel}")
                            await asyncio.sleep(1.5)
                            return True
                    except:
                        pass
            return False

        async def _check_save_status(current_wf):
            """Read SAP status bar and infer save outcome."""
            try:
                wf = await get_webgui_frame(page) or current_wf
                status = await get_status_bar_message(wf)
                if not status:
                    return None

                text = (status.get("text") or "").strip()
                kind = (status.get("type") or "").lower()
                if not text:
                    return None

                text_l = text.lower()
                if any(
                    k in text_l
                    for k in [
                        "saved",
                        "already saved",
                        "data was saved",
                        "data has been saved",
                    ]
                ):
                    LOGGER.info(f"  ✔ Save status: {text}")
                    return True
                # Common validation on this SSCUI when Bill/ex is selected but Sp.G/L indicator is empty
                if "specify sp.g/l ind" in text_l or "bill of exch" in text_l:
                    LOGGER.error(
                        "  ❌ Save validation failed: "
                        f"{text}. Set 'SpGL_Ind_BEx' (when Bill/ex is selected)."
                    )
                    return False
                if kind == "error":
                    LOGGER.error(f"  ❌ Save error status: {text}")
                    return False
                LOGGER.info(f"  📋 Status after save attempt: {text}")
                return None
            except Exception:
                return None

        async def _save_with_fallback(
            current_wf, prefer_click_save_first: bool = False
        ):
            """
            Save strategy:
            1) Trigger save (button-first or Ctrl+S-first)
            2) Handle dialogs/green tick + status check
            3) Retry with alternate trigger if needed
            """

            async def _run_single_save_attempt(use_button_trigger: bool):
                if use_button_trigger:
                    triggered = await _click_save_button_anywhere()
                    if not triggered:
                        return None
                else:
                    await page.keyboard.press("Control+s")
                    await asyncio.sleep(1.5)

                dialog_confirmed = await _confirm_save_dialogs()
                status_outcome = await _check_save_status(current_wf)

                if status_outcome is True:
                    return True
                if status_outcome is False:
                    return False

                # SAP does not always show explicit "saved" text. If dialog/green-tick flow completed
                # and no error status is present, treat as successful submission.
                if dialog_confirmed:
                    await asyncio.sleep(1.2)
                    status_outcome = await _check_save_status(current_wf)
                    if status_outcome is False:
                        return False
                    LOGGER.info("  ✔ Save dialog flow confirmed (green tick/OK).")
                    return True
                return None

            try:
                body = current_wf.locator("body").first
                if await body.count() > 0:
                    await body.click(force=True)
                    await asyncio.sleep(0.2)
            except:
                pass

            attempt_order = [True, False] if prefer_click_save_first else [False, True]
            for use_button in attempt_order:
                result = await _run_single_save_attempt(use_button)
                if result is not None:
                    return result

            LOGGER.warning("  ⚠ Save could not be confirmed for this SSCUI.")
            return False

        async def _ensure_edit_mode(current_wf):
            """Ensure detail screen is in edit mode (Edit button can exist outside WebGUI iframe)."""
            desc_inputs = await _find_labeled_inputs(current_wf, "Description")
            if any(not f.get("readOnly") for f in desc_inputs):
                return True

            edit_selectors = [
                "button:has-text('Edit')",
                "ui5-button:has-text('Edit')",
                "[title='Edit']",
                "[title*='Edit']",
                "[aria-label*='Edit']",
                ".sapMBtn:has-text('Edit')",
                "[role='button']:has-text('Edit')",
            ]

            for root in [page, *page.frames]:
                for sel in edit_selectors:
                    try:
                        btn = root.locator(sel).first
                        if await btn.is_visible(timeout=500):
                            await btn.click(force=True)
                            LOGGER.info(
                                "  ✔ Clicked 'Edit' button — switched to Edit mode."
                            )
                            await asyncio.sleep(2.5)
                            refreshed = await get_webgui_frame(page) or current_wf
                            desc_inputs = await _find_labeled_inputs(
                                refreshed, "Description"
                            )
                            editable = any(not f.get("readOnly") for f in desc_inputs)
                            if editable:
                                return True
                            LOGGER.warning(
                                "  ⚠ Edit clicked, but fields still appear read-only."
                            )
                            return False
                    except:
                        pass

            # Fallback: try F2 once
            try:
                await page.keyboard.press("F2")
                await asyncio.sleep(1)
            except:
                pass
            refreshed = await get_webgui_frame(page) or current_wf
            desc_inputs = await _find_labeled_inputs(refreshed, "Description")
            editable = any(not f.get("readOnly") for f in desc_inputs)
            if editable:
                LOGGER.info("  ✔ Edit mode enabled via F2 fallback.")
                return True

            LOGGER.warning(
                "  ⚠ Could not confirm Edit mode — proceeding with best effort."
            )
            return False

        async def _position_to_country_method(wf, country_code, method_code):
            """Use Position... popup to navigate to a Country/Payment Method combination."""
            try:
                pos_btn = wf.locator("text=/Position/i").first
                await pos_btn.scroll_into_view_if_needed()
                await pos_btn.click(force=True, timeout=10000)
            except:
                await wf.evaluate("""() => {
                    const btns = document.querySelectorAll('button, span, div');
                    for (const b of btns) {
                        if ((b.textContent || '').trim().match(/^Position/i)) {
                            b.click();
                            return;
                        }
                    }
                }""")
            await asyncio.sleep(1.5)
            await page.keyboard.type(country_code, delay=30)
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.2)
            await page.keyboard.type(method_code, delay=30)
            await page.keyboard.press("Enter")
            await asyncio.sleep(2.5)

        async def _is_country_method_visible(wf, country_code, method_code):
            """Return True only when an exact Country/Payment Method row is visible in the list."""
            return await wf.evaluate(
                """([c, p]) => {
                const normalize = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                const tokenize = (row) => {
                    const text = normalize(row.innerText || row.textContent || '');
                    return text ? text.split(/\\s+/).map(t => t.trim()).filter(Boolean) : [];
                };

                const rows = Array.from(document.querySelectorAll('tr, div[role="row"]'));
                for (const row of rows) {
                    const tokens = tokenize(row);
                    if (tokens.includes(c) && tokens.includes(p)) {
                        return true;
                    }
                }
                return false;
            }""",
                [country_code, method_code],
            )

        async def _open_country_method_detail(wf, country_code, method_code):
            """Open the exact Country/Payment Method row and return True on success."""
            opened = await wf.evaluate(
                """([c, p]) => {
                const normalize = (s) => (s || '').replace(/\\s+/g, ' ').trim();
                const tokenize = (row) => {
                    const text = normalize(row.innerText || row.textContent || '');
                    return text ? text.split(/\\s+/).map(t => t.trim()).filter(Boolean) : [];
                };

                const rows = Array.from(document.querySelectorAll('tr, div[role="row"]'));
                for (const row of rows) {
                    const tokens = tokenize(row);
                    if (!(tokens.includes(c) && tokens.includes(p))) continue;

                    row.scrollIntoView({behavior: 'instant', block: 'center'});
                    const target = row.querySelector('td, span, div[role="gridcell"], a') || row;
                    target.dispatchEvent(new MouseEvent('dblclick', {
                        bubbles: true,
                        cancelable: true,
                        view: window
                    }));
                    return true;
                }
                return false;
            }""",
                [country_code, method_code],
            )

            if opened:
                await asyncio.sleep(5)
            return opened

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page, EMAIL, PASSWORD)
            if await check_and_abort_if_locked(page):
                return

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WebGUI frame not found!")
                return

            await asyncio.sleep(5)  # Stabilization

            for idx, target in enumerate(targets, start=1):
                country = str(target.get("Country_Region", "")).strip()
                pymt_meth = str(target.get("Pymt_Meth", "")).strip()
                description = str(target.get("Description", "")).strip()
                description_2_raw = target.get("Description_2")
                description_2 = (
                    description
                    if description_2_raw is None
                    else str(description_2_raw).strip()
                )

                # Radio groups: support both string and dict format
                # Dict format: {"Outgoing payments": True, "Incoming payments": False} → picks the True key
                pmt_for_raw = target.get("Payment_Method_For", "")
                if isinstance(pmt_for_raw, dict):
                    pmt_for = next((k for k, v in pmt_for_raw.items() if v is True), "")
                else:
                    pmt_for = pmt_for_raw

                pmt_class_raw = target.get("Payment_Method_Classification", "")
                if isinstance(pmt_class_raw, dict):
                    pmt_class = next(
                        (k for k, v in pmt_class_raw.items() if v is True), ""
                    )
                else:
                    pmt_class = pmt_class_raw

                # Checkboxes — Payment method section
                post_office = target.get("Post_Office_Curr_Acct_Method")
                personnel = target.get("Allowed_For_Personnel_Payments")
                bill_before_due = target.get("Create_Bill_Exch_Before_Due_Date")
                bill_accepted = target.get("Bill_Of_Exch_Accepted")
                isr_procedure = target.get("ISR_Payment_Procedure")
                eu_transfer = target.get("EU_Internal_Transfer")

                # Checkboxes — Required master record specifications
                street_po = target.get("Street_PO_Box")
                bank_details = target.get("Bank_Details")
                account_number_req = target.get("Account_Number_Required")
                iban_req = target.get("IBAN_Required")
                swift_req = target.get("SWIFT_Code_Required")
                alias_req = target.get("Alias_Required")
                collection_auth = target.get("Collection_Authorization")
                sepa_mandate = target.get("SEPA_Mandate_Required")

                # Posting Details
                doc_type = target.get("Document_Type_For_Payment", "")
                clearing_doc_type = target.get("Clearing_Document_Type", "")
                spgl_bex = target.get("SpGL_Ind_BEx", "")
                spgl_sf = target.get("SpGL_Ind_SF", "")
                payment_order_only = target.get("Payment_Order_Only")

                # Payment medium
                fmt = target.get("Format", "")
                fmt_supplement = target.get("Format_Supplement", "")

                LOGGER.info(
                    f"[{idx}/{len(targets)}] Processing Country={country}, Pymt Meth={pymt_meth}"
                )

                # ── Step 1: Position search to find existing entry ──
                await _position_to_country_method(webgui_frame, country, pymt_meth)

                # ── Step 2: Check if entry exists, open detail or create new ──
                is_new_entry = False
                row_found = await _is_country_method_visible(
                    webgui_frame, country, pymt_meth
                )

                if row_found:
                    LOGGER.info(
                        f"  Exact row {country}/{pymt_meth} found. Opening detail view..."
                    )
                    opened = await _open_country_method_detail(
                        webgui_frame, country, pymt_meth
                    )
                    if not opened:
                        LOGGER.warning(
                            f"  Exact row {country}/{pymt_meth} could not be opened. "
                            "Proceeding with New Entries instead."
                        )
                        row_found = False

                if not row_found:
                    LOGGER.info(
                        f"  Row {country}/{pymt_meth} NOT found. Creating new entry..."
                    )

                    description = description or "Bank Transfer"
                    description_2 = description_2 or "Bank Transferr"
                    pmt_for = "Outgoing payments"
                    pmt_class = "Bank transf"

                    post_office = False
                    personnel = False
                    bill_before_due = False
                    bill_accepted = False
                    isr_procedure = False
                    eu_transfer = False

                    street_po = False
                    bank_details = True
                    account_number_req = True
                    iban_req = True
                    swift_req = True
                    alias_req = False
                    collection_auth = False
                    sepa_mandate = False

                    doc_type = "ZP"
                    clearing_doc_type = "ZV"
                    spgl_bex = ""
                    spgl_sf = ""
                    payment_order_only = False

                    LOGGER.info(
                        f"  [New Entry] Using Country={country}, "
                        f"Pymt Meth={pymt_meth}, "
                        f"Description1='{description}', Description2='{description_2}'"
                    )

                    await _click_sap_button(webgui_frame, "New Entries")
                    await asyncio.sleep(4)
                    is_new_entry = True

                # Re-acquire frame after navigation
                webgui_frame = await get_webgui_frame(page) or webgui_frame
                await asyncio.sleep(2)

                # ── Switch to Edit mode if in Display mode ──
                if not is_new_entry:
                    await _ensure_edit_mode(webgui_frame)
                    webgui_frame = await get_webgui_frame(page) or webgui_frame

                # ── Step 3: Handle New Entry dialog ("Another entry" popup) ──
                if is_new_entry:
                    LOGGER.info(
                        f"  [New Entry] Filling Country/Region and Pymt Meth..."
                    )

                    # Check for the "Another entry" dialog popup
                    dialog_found = False
                    try:
                        # Look for dialog with Country/Reg. and Pymt Meth fields
                        dialog = webgui_frame.locator("text=/Another entry/i").first
                        if await dialog.is_visible(timeout=5000):
                            LOGGER.info("  'Another entry' dialog detected.")
                            dialog_found = True
                    except:
                        pass

                    if dialog_found:
                        # Fill Country/Reg. field in dialog
                        country_input = await _click_first_input(
                            webgui_frame, "Country/Reg"
                        )
                        if country_input:
                            await _type_field(country)
                            LOGGER.info(f"    ✔ Country/Reg. = '{country}'")
                        else:
                            LOGGER.warning(
                                "    Could not find Country/Reg. input, trying Tab..."
                            )
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.3)
                            await _type_field(country)

                        # Tab to Pymt Meth field
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.3)
                        await _type_field(pymt_meth)
                        LOGGER.info(f"    ✔ Pymt Meth. = '{pymt_meth}'")

                        # Click Continue button
                        try:
                            cont_btn = webgui_frame.locator("text=/Continue/i").first
                            if await cont_btn.is_visible(timeout=3000):
                                await cont_btn.click(force=True)
                                LOGGER.info("    ✔ 'Continue' clicked")
                            else:
                                await page.keyboard.press("Enter")
                                LOGGER.info("    ✔ Enter pressed (Continue fallback)")
                        except:
                            await page.keyboard.press("Enter")
                        await asyncio.sleep(5)
                    else:
                        # No dialog — anchor on first input field directly
                        LOGGER.info("  No dialog popup. Anchoring on first input...")
                        clicked = await _click_first_input(webgui_frame, "Country/Reg")
                        if clicked:
                            await _type_field(country)
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.3)
                            await _type_field(pymt_meth)
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(4)
                        else:
                            # Tab-based fallback
                            for _ in range(3):
                                await page.keyboard.press("Tab")
                                await asyncio.sleep(0.3)
                            await _type_field(country)
                            await page.keyboard.press("Tab")
                            await asyncio.sleep(0.3)
                            await _type_field(pymt_meth)
                            await page.keyboard.press("Enter")
                            await asyncio.sleep(4)

                    # Handle any confirmation dialogs
                    await handle_sap_confirmation_dialogs(page)
                    await asyncio.sleep(2)

                    # Re-acquire frame
                    webgui_frame = await get_webgui_frame(page) or webgui_frame

                LOGGER.info("  Detail view ready.")

                # ═══════════════════════════════════════════════
                #  DETAIL VIEW — Fill fields
                # ═══════════════════════════════════════════════

                # ── Description fields (SAP shows two Description lines) ──
                if description or description_2_raw is not None:
                    LOGGER.info("  [Detail] Filling Description...")
                    desc_candidates = await _find_labeled_inputs(
                        webgui_frame, "Description"
                    )
                    if not desc_candidates:
                        LOGGER.warning("    ✘ Could not find Description inputs")
                    else:
                        await _set_input_by_id(
                            webgui_frame,
                            desc_candidates[0]["id"],
                            description,
                            "Description[1]",
                            commit_key="Tab",
                        )
                        if len(desc_candidates) > 1:
                            await _set_input_by_id(
                                webgui_frame,
                                desc_candidates[1]["id"],
                                description_2,
                                "Description[2]",
                                commit_key="Tab",
                            )
                        else:
                            LOGGER.warning(
                                "    ⚠ Only one Description field detected on this screen."
                            )

                # ── Payment method for (radio group) ──
                if pmt_for:
                    LOGGER.info(f"  [Detail] Setting Payment method for: '{pmt_for}'")
                    await _click_radio(webgui_frame, pmt_for)

                # ── Payment method classification (radio group) ──
                if pmt_class:
                    LOGGER.info(
                        f"  [Detail] Setting Payment method classification: '{pmt_class}'"
                    )
                    await _click_radio(webgui_frame, pmt_class)

                # ── Checkboxes (Payment method section) ──
                LOGGER.info("  [Detail] Processing payment method checkboxes...")
                checkbox_map = [
                    ("Post office curr.acct method", post_office),
                    ("Allowed for personnel payments", personnel),
                    ("Create bill/exch.before due date", bill_before_due),
                    ("Bill of exch. accepted", bill_accepted),
                    ("ISR Payment Procedure", isr_procedure),
                    ("EU Internal Transfer", eu_transfer),
                ]
                for label, state in checkbox_map:
                    await _set_checkbox_by_label(webgui_frame, label, state)

                # ── Required master record specifications (checkboxes) ──
                LOGGER.info("  [Detail] Required master record specifications...")
                master_record_checkboxes = [
                    ("Street,P.O.box or P.O.box pst code", street_po),
                    ("Bank details", bank_details),
                    ("Account Number Required", account_number_req),
                    ("IBAN Required", iban_req),
                    ("SWIFT Code Required", swift_req),
                    ("Alias Required", alias_req),
                    ("Collection authorization", collection_auth),
                    ("SEPA Mandate Required", sepa_mandate),
                ]
                for label, state in master_record_checkboxes:
                    await _set_checkbox_by_label(webgui_frame, label, state)

                # ── Posting Details ──
                LOGGER.info("  [Detail] Posting Details...")

                # Scroll down to make Posting Details visible
                await page.keyboard.press("PageDown")
                await asyncio.sleep(1)

                doc_input_id = None
                clearing_input_id = None

                if doc_type:
                    LOGGER.info(
                        f"  [Detail] Filling Document Type for Payment: '{doc_type}'"
                    )
                    doc_input_id = await _set_labeled_input(
                        webgui_frame,
                        "Document Type for Payment",
                        doc_type,
                        occurrence=0,
                        field_name="Document Type for Payment",
                        commit_key="Tab",
                    )

                if clearing_doc_type:
                    LOGGER.info(
                        f"  [Detail] Filling Clearing Document Type: '{clearing_doc_type}'"
                    )
                    clearing_input_id = await _set_labeled_input(
                        webgui_frame,
                        "Clearing Document Type",
                        clearing_doc_type,
                        occurrence=0,
                        field_name="Clearing Document Type",
                        commit_key="Tab",
                        exclude_ids=[doc_input_id] if doc_input_id else None,
                    )

                # Diagnostic guard: ensure both posting fields did not resolve to same control
                if (
                    doc_input_id
                    and clearing_input_id
                    and doc_input_id == clearing_input_id
                ):
                    LOGGER.error(
                        "    ✘ Posting Details mapping conflict: both fields resolved "
                        f"to same input id={doc_input_id}"
                    )
                elif doc_input_id or clearing_input_id:
                    posting_values = await webgui_frame.evaluate(
                        """(ids) => {
                        const readVal = (id) => {
                            if (!id) return null;
                            const el = document.getElementById(id);
                            if (!el) return null;
                            return (el.value || '').trim();
                        };
                        return {
                            doc: readVal(ids.doc),
                            clearing: readVal(ids.clearing)
                        };
                    }""",
                        {"doc": doc_input_id, "clearing": clearing_input_id},
                    )
                    LOGGER.info(
                        "    ↳ Posting Details current values: "
                        f"Document Type='{posting_values.get('doc')}', "
                        f"Clearing='{posting_values.get('clearing')}'"
                    )

                if spgl_bex:
                    LOGGER.info(f"  [Detail] Filling Sp.G/L Ind.B/Ex.: '{spgl_bex}'")
                    clicked = await _click_first_input(webgui_frame, "Sp.G/L Ind.B/Ex")
                    if clicked:
                        await _type_field(spgl_bex)
                        LOGGER.info(f"    ✔ 'Sp.G/L Ind.B/Ex.' = '{spgl_bex}'")
                    else:
                        LOGGER.warning(f"    ✘ Could not find Sp.G/L Ind.B/Ex. input")

                if spgl_sf:
                    LOGGER.info(f"  [Detail] Filling Sp.G/L Ind. for SF: '{spgl_sf}'")
                    clicked = await _click_first_input(
                        webgui_frame, "Sp.G/L Ind. for SF"
                    )
                    if clicked:
                        await _type_field(spgl_sf)
                        LOGGER.info(f"    ✔ 'Sp.G/L Ind. for SF' = '{spgl_sf}'")
                    else:
                        LOGGER.warning(f"    ✘ Could not find Sp.G/L Ind. for SF input")

                if payment_order_only is not None:
                    await _set_checkbox_by_label(
                        webgui_frame, "Payment Order Only", payment_order_only
                    )

                # ── Payment medium ──
                if fmt or fmt_supplement:
                    LOGGER.info("  [Detail] Payment medium section...")

                if fmt:
                    LOGGER.info(f"  [Detail] Filling Format: '{fmt}'")
                    clicked = await _click_first_input(webgui_frame, "Format")
                    if clicked:
                        await _type_field(fmt)
                        LOGGER.info(f"    ✔ 'Format' = '{fmt}'")
                    else:
                        LOGGER.warning(f"    ✘ Could not find Format input")

                if fmt_supplement:
                    LOGGER.info(
                        f"  [Detail] Filling Format supplement: '{fmt_supplement}'"
                    )
                    clicked = await _click_first_input(
                        webgui_frame, "Format supplement"
                    )
                    if clicked:
                        await _type_field(fmt_supplement)
                        LOGGER.info(f"    ✔ 'Format supplement' = '{fmt_supplement}'")
                    else:
                        LOGGER.warning(f"    ✘ Could not find Format supplement input")

                LOGGER.info(
                    f"  ✔ Country={country}, Pymt Meth={pymt_meth} — all fields processed."
                )

                # ═══════════════════════════════════════════════
                #  SAVE FLOW (as requested):
                #  1) Save on detail screen
                #  2) Back to list view
                #  3) Save again from list view
                #  4) Click green tick / OK
                #  5) Exit
                #  6) Logout
                # ═══════════════════════════════════════════════
                if is_new_entry:
                    LOGGER.info(
                        "  [New Entry] Saving detail screen before going back..."
                    )
                    detail_save_confirmed = await _save_with_fallback(
                        webgui_frame, prefer_click_save_first=True
                    )
                    if not detail_save_confirmed:
                        LOGGER.warning(
                            "  ⚠ Detail-screen save was not confirmed — check this record manually."
                        )
                    await _confirm_save_dialogs()
                    await handle_sap_confirmation_dialogs(page)
                    await asyncio.sleep(1)
                    webgui_frame = await get_webgui_frame(page) or webgui_frame

                LOGGER.info("  Returning to list view before final save...")
                await page.keyboard.press("F3")
                await asyncio.sleep(2)
                await _confirm_save_dialogs()
                await handle_sap_confirmation_dialogs(page)
                await asyncio.sleep(1)
                webgui_frame = await get_webgui_frame(page) or webgui_frame

                LOGGER.info("  Saving changes from list view...")
                save_confirmed = await _save_with_fallback(
                    webgui_frame, prefer_click_save_first=True
                )
                if not save_confirmed:
                    LOGGER.warning(
                        "  ⚠ Save was not confirmed — check this record manually."
                    )
                else:
                    LOGGER.info("  ✔ List-view save confirmed.")

                LOGGER.info(
                    "  Confirming any green tick / OK dialog after final save..."
                )
                await _confirm_save_dialogs()
                await handle_sap_confirmation_dialogs(page)
                await asyncio.sleep(1)

                LOGGER.info("  Exiting transaction...")
                exited = await _click_exit_button_anywhere()
                if not exited:
                    await page.keyboard.press("F3")
                    LOGGER.info("  Exit button not found — used F3 fallback.")
                await asyncio.sleep(2)
                # Handle any "Save changes?" dialog that appears on F3
                for frame in page.frames:
                    try:
                        yes_btn = frame.locator(
                            "button:has-text('Yes'), [title='Yes']"
                        ).first
                        if await yes_btn.is_visible(timeout=1000):
                            await yes_btn.click(force=True)
                            LOGGER.info("  ✔ Clicked 'Yes' on F3 exit dialog.")
                            await asyncio.sleep(2)
                            break
                        no_btn = frame.locator(
                            "button:has-text('No'), [title='No']"
                        ).first
                        if await no_btn.is_visible(timeout=500):
                            await no_btn.click(force=True)
                            LOGGER.info(
                                "  ✔ Clicked 'No' on F3 exit dialog (already saved)."
                            )
                            await asyncio.sleep(2)
                            break
                    except:
                        pass
                LOGGER.info(f"  ✔ Country {country} / Payt {pymt_meth} complete.")

            LOGGER.info("Logging out of SAP...")
            await smart_logout(page)
            LOGGER.info("DONE.")
        except Exception as e:
            LOGGER.error(f"Error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # ── Proper SAP cleanup to release locks ──
            try:
                LOGGER.info("  🔒 Releasing SAP lock and cleaning up session...")
                # Handle any pending "Save changes?" dialog first
                for frame in page.frames:
                    try:
                        no_btn = frame.locator(
                            "button:has-text('No'), [title='No']"
                        ).first
                        if await no_btn.is_visible(timeout=500):
                            await no_btn.click(force=True)
                            await asyncio.sleep(1)
                            break
                    except:
                        pass
                # Navigate to SAP logoff URL to fully release the session
                try:
                    await page.goto(
                        "https://my401292.s4hana.cloud.sap/sap/public/bc/icf/logoff?sap-client=100",
                        timeout=10000,
                    )
                    LOGGER.info("  ✔ SAP session logged off.")
                except:
                    pass
            except Exception as cleanup_err:
                LOGGER.warning(f"  Cleanup warning: {cleanup_err}")
            finally:
                await browser.close()
                LOGGER.info("  ✔ Browser closed.")


async def Maintain_Terms_of_Payment_102934(targets: list[dict]):

    url = "https://my401292.s4hana.cloud.sap/ui#IMGActivity-execute?IMGActivity=SIMG_XXMENUORFBOBB8&CustomizingObject=V_T052&CustomizingObjectType=V&CustomizingProject=&CustomizingTransaction=S_ALR_87003082&Type=SSCUI"

    async with async_playwright() as p:
        browser = await launch_sap_browser(p)
        context = await browser.new_context(viewport={"width": 1280, "height": 800})
        page = await new_page_in_front(context)

        # ── Local helpers ──
        async def _type_field(value):
            """Select all → delete → type new value."""
            await page.keyboard.down("Control")
            await page.keyboard.press("a")
            await page.keyboard.up("Control")
            await asyncio.sleep(0.1)
            await page.keyboard.press("Delete")
            await asyncio.sleep(0.1)
            if value:
                await page.keyboard.type(str(value), delay=0)
            await asyncio.sleep(0.2)

        async def _tab_and_type(label, value):
            """Tab to next field and type value."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(0.2)
            LOGGER.info(f"    {label} → '{value}'")
            await _type_field(value)

        async def _set_checkbox_via_tab(field_name, target_state, wf):
            """Tab to checkbox, detect state, toggle if needed."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(1)
            if target_state is None:
                LOGGER.info(f"    Checkbox '{field_name}': skipped")
                return
            JS_CHK = """
            () => {
                const el = document.activeElement;
                if (!el) return null;
                const checkLs = (n) => {
                    if (!n) return null;
                    const raw = n.getAttribute('lsdata');
                    if (raw) {
                        try {
                            const d = JSON.parse(raw.replace(/'/g, '"'));
                            if (d['1'] === true || d['1'] === '1' || d['1'] === 1) return true;
                            if (d['1'] === false || d['1'] === '0' || d['1'] === 0 || d['1'] === '') return false;
                        } catch(e) {}
                    }
                    return null;
                };
                let s = checkLs(el);
                if (s !== null) return s;
                s = checkLs(el.parentElement);
                if (s !== null) return s;
                if (el.tagName === 'INPUT' && el.type === 'checkbox') return el.checked;
                const inp = el.querySelector('input[type="checkbox"]');
                if (inp) return inp.checked;
                const aria = el.getAttribute('aria-checked');
                if (aria === 'true') return true;
                if (aria === 'false') return false;
                const cls = (el.className || '') + ' ' + (el.parentElement?.className || '');
                if (cls.includes('SAPBChk-Chk')) return true;
                if (cls.includes('SAPBChk-Uchk') || cls.includes('SAPBChk')) return false;
                return null;
            }
            """
            try:
                state = await wf.evaluate(JS_CHK)
                LOGGER.info(
                    f"    Checkbox '{field_name}': current={state}, target={target_state}"
                )
                if state is not None:
                    if state != target_state:
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.5)
                        LOGGER.info(f"      → Toggled")
                    else:
                        LOGGER.info(f"      → Already correct")
                else:
                    LOGGER.warning(f"      ! Detection failed — fallback")
                    if target_state:
                        await page.keyboard.press("Space")
                        await asyncio.sleep(0.5)
            except Exception as e:
                LOGGER.warning(f"      ! Error: {e}")
                if target_state:
                    await page.keyboard.press("Space")
                    await asyncio.sleep(0.5)

        async def _set_radio_group(label, desired_option, wf):
            """Tab to radio group, select desired option by arrow keys."""
            await page.keyboard.press("Tab")
            await asyncio.sleep(1)
            LOGGER.info(f"    Radio '{label}' → '{desired_option}'")
            radio_map = {
                "No Default": 0,
                "Posting Date": 1,
                "Document Date": 2,
                "Entry Date": 3,
            }
            presses = radio_map.get(desired_option, 0)
            for _ in range(presses):
                await page.keyboard.press("ArrowDown")
                await asyncio.sleep(0.3)

        async def _click_first_input_near(wf, label_text):
            """Click the first input near a label."""
            return await wf.evaluate(
                """(labelText) => {
                const allElements = Array.from(document.querySelectorAll('span, label, td'));
                for (const el of allElements) {
                    const t = el.textContent.trim();
                    if (t.includes(labelText) && t.length < labelText.length + 15) {
                        let parent = el.parentElement;
                        for (let i = 0; i < 6; i++) {
                            if (!parent) break;
                            const inp = parent.querySelector('input[type="text"], input:not([type]), input[type="number"]');
                            if (inp && !inp.disabled && !inp.readOnly) {
                                inp.scrollIntoView({behavior: 'instant', block: 'center'});
                                inp.focus();
                                inp.click();
                                return true;
                            }
                            parent = parent.parentElement;
                        }
                    }
                }
                return false;
            }""",
                label_text,
            )

        try:
            LOGGER.info(f"OPENING: {url}")
            await page.goto(url, wait_until="commit", timeout=60000)
            await login(page)
            if await check_and_abort_if_locked(page):
                return
            await asyncio.sleep(1)

            webgui_frame = await get_webgui_frame(page)
            if not webgui_frame:
                LOGGER.error("WEBGUI IFRAME NOT FOUND!")
                return

            LOGGER.info(f"Processing {len(targets)} payment term target(s)...")

            for idx, target in enumerate(targets, start=1):
                pmt_term = str(target.get("Payment_Terms", "")).strip()
                day_limit = str(target.get("Day_Limit", "0")).strip()
                sales_text = target.get("Sales_Text", "")
                own_expl = target.get("Own_Explanation", "")
                acct_type = target.get("Account_Type", {})
                baseline_calc = target.get("Baseline_Date_Calculation", {})
                default_base = target.get("Default_Baseline_Date", "No Default")
                pmnt_block = target.get("Pmnt_Block", {})
                pmt_method = target.get("Payment_Method", {})
                installment = target.get("Installment_Payments", False)
                rec_entries = target.get("Rec_Entries_Supplement", False)
                pmt_lines = target.get("Payment_Terms_Lines", [{}, {}, {}])
                explanations = target.get("Explanations", "")
                hide_entry = target.get("Hide_Entry_in_Input_Help", False)

                LOGGER.info(
                    f"[{idx}/{len(targets)}] ═══ Payment Term: {pmt_term}, Day Limit: {day_limit} ═══"
                )

                # ── Step 1: Use Position... to search ──
                is_new_entry = False
                try:
                    pos_btn = webgui_frame.locator("text=/Position/i").first
                    if await pos_btn.is_visible(timeout=5000):
                        await pos_btn.click(force=True)
                        await asyncio.sleep(1.5)
                        await page.keyboard.type(pmt_term, delay=0)
                        await page.keyboard.press("Enter")
                        LOGGER.info(f"  Position search for '{pmt_term}'")
                        await asyncio.sleep(1)
                except Exception as e:
                    LOGGER.warning(f"  Position search failed: {e}")

                # ── Step 2: Check if term exists in list ──
                row_found = await webgui_frame.evaluate(
                    """(termId) => {
                    const allEls = document.querySelectorAll('span, td, div, a');
                    for (const el of allEls) {
                        const t = (el.textContent || '').trim();
                        if (t === termId) {
                            el.scrollIntoView({behavior: 'instant', block: 'center'});
                            return true;
                        }
                    }
                    return false;
                }""",
                    pmt_term,
                )

                if row_found:
                    LOGGER.info(f"  ✔ Term '{pmt_term}' FOUND. Opening detail view...")
                    term_el = webgui_frame.locator(f"text='{pmt_term}'").first
                    try:
                        await term_el.click(force=True)
                        await asyncio.sleep(1)
                    except:
                        pass

                    # Try clicking "Details" button to open detail view
                    detail_opened = False
                    try:
                        det_btn = webgui_frame.locator("text=/Details/i").first
                        if await det_btn.is_visible(timeout=3000):
                            await det_btn.click(force=True)
                            detail_opened = True
                            LOGGER.info("  ✔ Details button clicked")
                            await asyncio.sleep(4)
                    except:
                        pass

                    if not detail_opened:
                        # Fallback: double-click on the term
                        try:
                            await term_el.dblclick(force=True, timeout=10000)
                            LOGGER.info("  ✔ Double-clicked term to open detail")
                            await asyncio.sleep(4)
                        except:
                            LOGGER.warning("  Could not open detail view")
                else:
                    LOGGER.info(
                        f"  ✖ Term '{pmt_term}' NOT found. Creating new entry..."
                    )
                    is_new_entry = True
                    new_clicked = False
                    for frame in page.frames:
                        try:
                            btn = frame.locator("text=/New Entries/i").first
                            if await btn.is_visible(timeout=5000):
                                await btn.click(force=True)
                                new_clicked = True
                                LOGGER.info("  ✔ 'New Entries' clicked.")
                                await asyncio.sleep(5)
                                break
                        except:
                            continue
                    if not new_clicked:
                        LOGGER.error("  Could not click 'New Entries'! Skipping.")
                        continue

                # ── Re-acquire frame after navigation ──
                webgui_frame = await get_webgui_frame(page) or webgui_frame
                await asyncio.sleep(2)

                # ═══════════════════════════════════════════════
                #  DETAIL FORM — Fill all fields
                # ═══════════════════════════════════════════════
                LOGGER.info("  [Detail] Filling form fields...")

                # Anchor on Payment Terms field
                anchored = await _click_first_input_near(webgui_frame, "Payment Terms")
                if not anchored:
                    LOGGER.warning(
                        "  Could not anchor on 'Payment Terms'. Trying Tab fallback."
                    )
                    for _ in range(3):
                        await page.keyboard.press("Tab")
                        await asyncio.sleep(0.3)
                await asyncio.sleep(0.5)

                # Field 1: Payment Terms
                LOGGER.info(f"  [1] Payment Terms → '{pmt_term}'")
                await _type_field(pmt_term)

                # Field 2: Sales text
                await _tab_and_type("Sales text", sales_text)

                # Field 3: Day Limit
                await _tab_and_type("Day Limit", day_limit)

                # Field 4: Own Explanation
                await _tab_and_type("Own Explanation", own_expl)

                # Field 5: Customer checkbox (Account type)
                await _set_checkbox_via_tab(
                    "Customer", acct_type.get("Customer"), webgui_frame
                )

                # Field 6: Supplier checkbox (Account type)
                await _set_checkbox_via_tab(
                    "Supplier", acct_type.get("Supplier"), webgui_frame
                )

                # Field 7: Fixed Day (Baseline date calculation)
                await _tab_and_type(
                    "Baseline Fixed Day", baseline_calc.get("Fixed_Day", "")
                )

                # Field 8: Additional Months (Baseline date calculation)
                await _tab_and_type(
                    "Baseline Additional Months",
                    baseline_calc.get("Additional_Months", ""),
                )

                # Field 9: Default for baseline date (radio group)
                await _set_radio_group(
                    "Default for baseline date", default_base, webgui_frame
                )

                # Field 10: Block key (Pmnt block)
                await _tab_and_type("Block key", pmnt_block.get("Block_Key", ""))

                # Field 11: Block key checkbox
                await _set_checkbox_via_tab(
                    "Block key checkbox",
                    pmnt_block.get("Block_Key_Checkbox"),
                    webgui_frame,
                )

                # Field 12: Payment Method
                await _tab_and_type("Payment Method", pmt_method.get("Method", ""))

                # Field 13: Payment Method checkbox
                await _set_checkbox_via_tab(
                    "Payment Method checkbox",
                    pmt_method.get("Method_Checkbox"),
                    webgui_frame,
                )

                # Field 14: Installment Payments checkbox
                await _set_checkbox_via_tab(
                    "Installment Payments", installment, webgui_frame
                )

                # Field 15: Rec. Entries: Supplement fm Master checkbox
                await _set_checkbox_via_tab(
                    "Rec. Entries Supplement", rec_entries, webgui_frame
                )

                # ── Payment Terms Lines (3 rows) ──
                LOGGER.info("  [Payment Terms Lines] Filling 3 rows...")

                line1 = pmt_lines[0] if len(pmt_lines) > 0 else {}
                await _tab_and_type("Line1 Percentage", line1.get("Percentage", ""))
                await _tab_and_type("Line1 No of Days", line1.get("No_of_Days", ""))
                await _tab_and_type("Line1 Fixed Day", line1.get("Fixed_Day", ""))
                await _tab_and_type(
                    "Line1 Additional Months", line1.get("Additional_Months", "")
                )

                line2 = pmt_lines[1] if len(pmt_lines) > 1 else {}
                await _tab_and_type("Line2 Percentage", line2.get("Percentage", ""))
                await _tab_and_type("Line2 No of Days", line2.get("No_of_Days", ""))
                await _tab_and_type("Line2 Fixed Day", line2.get("Fixed_Day", ""))
                await _tab_and_type(
                    "Line2 Additional Months", line2.get("Additional_Months", "")
                )

                line3 = pmt_lines[2] if len(pmt_lines) > 2 else {}
                await _tab_and_type("Line3 No of Days", line3.get("No_of_Days", ""))
                await _tab_and_type("Line3 Fixed Day", line3.get("Fixed_Day", ""))
                await _tab_and_type(
                    "Line3 Additional Months", line3.get("Additional_Months", "")
                )

                # Field: Explanations text
                await _tab_and_type("Explanations", explanations)

                # Field: Hide Entry in Input Help checkbox
                await _set_checkbox_via_tab(
                    "Hide Entry in Input Help", hide_entry, webgui_frame
                )

                LOGGER.info(f"  ✅ All fields filled for Term '{pmt_term}'")

                # ── Validate with Enter ──
                await page.keyboard.press("Enter")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page)
                await asyncio.sleep(1)

                # ── Save ──
                LOGGER.info("  Saving...")
                await page.keyboard.press("Control+s")
                await asyncio.sleep(3)
                await handle_sap_confirmation_dialogs(page)
                await asyncio.sleep(2)

                # Check status bar
                webgui_frame = await get_webgui_frame(page) or webgui_frame
                status = await get_status_bar_message(webgui_frame)
                if status and status.get("type") == "error":
                    err_text = status.get("text", "Unknown SAP error after save")
                    LOGGER.error(f"  ❌ SAP save error for Term '{pmt_term}': {err_text}")
                    raise RuntimeError(
                        f"SAP save rejected for Term '{pmt_term}': {err_text}"
                    )
                elif status:
                    LOGGER.info(f"  📋 Status: {status.get('text', '')}")
                    await page.keyboard.press("F3")
                    await asyncio.sleep(3)

                LOGGER.info(f"  ✔ Payment Term '{pmt_term}' complete.")

            LOGGER.info("DONE with 102934.")

        except Exception as e:
            LOGGER.error(f"FATAL ERROR in 102934: {e}", exc_info=True)
        finally:
            try:
                # Handle any pending save dialog with No (discard)
                for frame in page.frames:
                    try:
                        no_btn = frame.locator(
                            "button:has-text('No'), [title='No']"
                        ).first
                        if await no_btn.is_visible(timeout=500):
                            await no_btn.click(force=True)
                            await asyncio.sleep(1)
                            break
                    except:
                        pass
                try:
                    await page.goto(
                        "https://my401292.s4hana.cloud.sap/sap/public/bc/icf/logoff?sap-client=100",
                        timeout=10000,
                    )
                    LOGGER.info("  ✔ SAP session logged off.")
                except:
                    pass
            except Exception as ce:
                LOGGER.warning(f"  Cleanup warning: {ce}")
            finally:
                if not page.is_closed():
                    await browser.close()
                LOGGER.info("  ✔ Browser closed.")


async def main():
    LOGGER.info("Starting SAP Configuration Tools Manual Test...")

    # ── 1. Company Code General Ledger View (106039) ──
    target_106039 = [
        {
            "CoCd": "1810",
            "Max_ex_dev": "100",
            "No_Exch_Rate_Diff": False,
            "Negative_Postings_Permitted": True,
            "Enable_Amount_Split": False,
        }
    ]
    await company_code_gl_view_106039(targets=target_106039)

    # ── 2. Create Company (106040) ──
    target_106040 = [{
        "Company": "1235",
        "Name": "Test Company sat infotech",
        "Country": "IN"
    }]
    await create_global_company_106040(targets=target_106040)

    # ── 3. Financial Accounting Automation (100297) ── #LINK NOT WORKING
    target_100297 = [{
        "area": "Financial Accounting",
        "subarea": "General Ledger Accounting",
        "process": "G/L Account Master Data",
        "chart_of_accounts": "YCOA",
        "transaction_keys": ["GBB", "PRD"]
    }]
    await execute_financial_accounting_automation_100297(targets=target_100297)

    # ── 4. Maintain Budget Availability Control Profile for Cost Centers (102781) ──
    target_102781 = [     #WORK ON ERRORS BEFORE CLICKING ON TICK BUTTON AFTER SAVE
        {
            "Profile": "CC003",
            "Availy Ctrl Type": "Cost Center",
            "Availy Prfl Name": "Cost Center Budget",
            "Time_Range": "Annual Budget",
            "Budget Currency Type": "Company Code Currency",
        }
    ]
    await Maintain_Budget_Availability_Control_Profile_for_Cost_Centers_102781(
        targets=target_102781
    )

    # ── 5. Assign Company Code (101631) ── WORKING
    target_101631 = [{
        "CoCd": "1810",
        "Company":"3000"
    }]
    await assign_company_code_101631(targets=target_101631)

    # ── 6. Maintain Budget Availability Control Profile for Projects (102413) ── NOT WORKING (FIX/ CATCH ERRORS AFTER SAVE)
    target_102413 = [{
        "Profile": "0001"
    }]
    await Maintain_Budget_Availability_Control_Profile_for_Projects_102413(targets=target_102413)

    # ── 7. Edit Tax Information for Company Codes (105675) ── WORKING
    target_105675 = [
        {
            "CoCd": "1810",
            "Tx_Crcy_Transl": "Exchange rate determined using document date",
            "VAT_Reg_No": "IN123456780",
            "Tax_Base_Net": False,
            "Tax_Reporting_Date": False,
            "Discount_Base_Net": False,
        },
    ]
    await edit_tax_information_for_company_codes_105675(targets=target_105675)

    # # ── 8. Define Parameter Sets (103635) ── WORKING
    target_103635 = [
        {
            "Parameter Set ID": "ZBA1234",
            "P Set Name": "Standard Parameter Set",
            "Posting Method": "One Journal Entry per Bundle",
            "Ass.Val.Dte to Acct": True,
            "Summarization": False,
            "ZeroSales Permitted": True,
            "Bundle Type": "No Bundling",
            "Number of Items": "100"
        }
    ]
    await Define_Parameter_Sets_103635(targets=target_103635)

    # ── 9. Define Dunning Block Reasons (102259) ── WORIKING
    target_102259 = [{
        "Block": "A",
        "Text" : "This is testing one "
    },
    {
        "Block": "B",
        "Text" : "This is testing two"

    },
    {
        "Block": "H",
        "Text" : "this is test 3"
        
    }
    ]
    await Define_Dunning_Block_Reasons_102259(targets=target_102259)
   
    #  ── 10. Setup Paying Company Codes (101001) ──   WORKING
    target_101001 = [{
            "CoCd": "1810",
        "Control Data": {
        "Minimum Amount for Incoming Payment": "500",
        "Incoming Payment Currency": "INR",
        "Minimum Amount for Outgoing Payment": "24002",
        "Outgoing Payment Currency": "INR",
        "No Exchange Rate Differences": True,
        "No Exch.Rate Diffs. (Part Payments)": False,
        "Separate Payment for Each Ref.": True,
        "Bill/Exch Pymt": True
    },
    "Specifications for SEPA Payments": {
        "Creditor Identification Number": ""
    },
    "Bill of Exchange Data": {
        "Create Bills of Exchange": {
        "One Bill of Exchange per Invoice": False,
        "One Bill of Exchange per Due Date": False,
        "One Bill of Exch. per Due Date Per.": True
        },
        "Bill of Exch.Due Date/Bill of Exch.Pmnt Requests for Incoming Payments": {
        "Latest Due Date in Days": "10",
        "Bill on Demand for Due Date up Until Days": "5"
        },
        "Bill of Exchange Due Date for Outgoing Payments": {
        "Earliest Due Date in Days": "5",
        "Latest Due Date in Days": "10"
        }
    }
    }
        ]
    await setup_paying_company_codes_101001(targets=target_101001) 

    # ── 11. Maintain Additional Parameters (102739) ── LOOK IN TO IT NOTHING TO DO HERE
    target_102739 = [{
        CoCd": "1810"
    }]
    await Maintain_Additional_Parameters_102739(targets=target_102739)

    ── 12. Set Up All Company Codes for Payment Transactions (101293) ── WORKKING
    targets = [
        {
            "CoCd": "1810",
            "Separate Payment per Business Area": False,
            "Pyt Meth Suppl": False,
            "Tolerance Days for Payable": "10",
            "Outgoing Pmnt with Cash Disc.From": "10",
            "Max.Cash Discount": False,
            "Vendors": {
                "Sp. G/L Transactions to Be Paid": "F1VX",
                "Sp. G/L Trans. for Exception List": "I"
            },
            "Customers": {
                "Sp. G/L Transactions to Be Paid": "1AGK",
                "Sp. G/L Trans. for Exception List": "T"
            }
        },
        {
            "CoCd": "9999",
            "Sending Company Code": "9999",
            "Paying Company Code": "9999",
            "Separate Payment per Business Area": True,
            "Pyt Meth Suppl": False,
            "Tolerance Days for Payable": "15",
            "Outgoing Pmnt with Cash Disc.From": "5",
            "Max.Cash Discount": True,
            "Vendors": {
                "Sp. G/L Transactions to Be Paid": "F1",
                "Sp. G/L Trans. for Exception List": "V"
            },
            "Customers": {
                "Sp. G/L Transactions to Be Paid": "1A",
                "Sp. G/L Trans. for Exception List": "T"
            }
        }
    ]
    await Set_Up_All_Company_Codes_for_Payment_Transactions_101293(targets=targets) #ADD ERROR DETECTION

    # ── 13. Set Up Bank Determination for Payment Transactions (101045) ──
    target_101045 = [{
        "Paying Company Code": "1810",
        "Ranking Order": [
            { "Housebank": ""},
            { "Housebank": ""},
        ],
        "Bank Accounts": [
            {"House Bank": ".", "Payment Method": ".", "Currency": ".", "Account ID": ".", "Bank Subaccount": ".","Clear.acct":".","Charge ind":".","Bus. area":"."},
            {"House Bank": ".", "Payment Method": ".", "Currency": ".",    "Account ID": ".", "Bank Subaccount": ".","Clear.acct":".","Charge ind":".","Bus. area":"."},
        ],
        "Bank Accounts (Enhanced)": [
            {"House Bank": ".", "Payment Method": ".", "Currency": ".", "Ranking Order": ".", "Account ID": ".", "Bank Subaccount": ".", "Clearing Account": ".", "Charge Ind.": ".","BusA":""},
            {"House Bank": ".", "Payment Method": ".", "Currency": ".", "Ranking Order": ".", "Account ID": ".", "Bank Subaccount": ".", "Clearing Account": ".", "Charge Ind.": ".","BusA":""},
        ],
        "Value Date": [
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".", "Days to Value Date": ".", "Company Name": "."},
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".", "Days to Value Date": ".", "Company Name": "."},
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".",    "Days to Value Date": ".", "Company Name": "."},
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".", "Days to Value Date": ".", "Company Name": "."},
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".", "Days to Value Date": ".", "Company Name": "."},
            {"Payment Method": ".", "House Bank": ".", "Account ID": ".", "Currency": ".", "Days to Value Date": ".", "Company Name": "."},
        ],
        "Expenses/Charges": [
            {"Charge ind":".","Amount Limits": ".","Currency":".","Charges 1":".","Charges 2":"."}
            
        ]
        
    }]
    await Set_Up_Bank_Determination_for_Payment_Transactions_101045(targets=target_101045)


    # ── 14. Define Account Determination for Bank Clearing Accounts (102803) ──
    target_102803 = [{
        "Paying Company Code": "1810",
        "Entries": [
            {"House Bk": "", "PM": "C", "Crcy": "", "Acct ID": "", "Bank Subacct": "11001050", "Charge Ind": "", "BusA": ""},
            {"House Bk": "", "PM": "F", "Crcy": "", "Acct ID": "", "Bank Subacct": "11001020", "Charge Ind": "", "BusA": ""},
            {"House Bk": "", "PM": "T", "Crcy": "", "Acct ID": "", "Bank Subacct": "11001020", "Charge Ind": "", "BusA": ""},
            {"House Bk": "", "PM": "T", "Crcy": "USD", "Acct ID": "", "Bank Subacct": "11001020", "Charge Ind": "", "BusA": ""}
        ]
    }]
    await Define_Account_Determination_for_Bank_Clearing_Accounts_102803(targets=target_102803)

#   # ── 15. Define Clearing Accounts for Receiving Bank for Account Transfer (102802) ──
    target_102802 = [
        {"CoCode": "", "House Bk": "", "Cntry/Reg.": "", "Payt Meth.": "", "Currency": "", "Account ID": "", "Clrg Acct": ""},
        {"CoCode": "", "House Bk": "", "Cntry/Reg.": "", "Payt Meth.": "", "Currency": "", "Account ID": "", "Clrg Acct": ""},
        {"CoCode": "", "House Bk": "", "Cntry/Reg.": "", "Payt Meth.": "", "Currency": "", "Account ID": "", "Clrg Acct": ""},
    ]
    await Define_Clearing_Accounts_for_Receiving_Bank_for_Account_Transfer_102802(targets=target_102802)

    # # ── 16. Maintenance of Company Code Data for a Payment Method (101044) ──
    target_101044 = [{
        "CoCode": "1810",
        "Payt Meth.": "C",
        "Data": {
            "Amount Limits": {"Minimum Amount": "", "Maximum Amount": "", "Distrib. Amount": ""},
            "Grouping of Items": {"Single Payment for Marked Item": True, "Payment per Due Day": False, "Extended Individual Payment": False},
            "Foreign Payments": {"Foreign Business Partner Allowed": True, "Foreign Currency Allowed": False, "Bank Abroad Allowed": False},
            "Bank Selection Control": {"No Optimization": False, "Optimize by Bank Group": True, "Optimize by Postal Code": False},
            "Forms": {"Payment Medium Type": "PDF", "Form for the Payment Medium": "FIN_FO_PAYM_CHECK_UK"},
            "Drawer on the form": {"label_text1": "Test Drawer 1", "label_text2": "Test Drawer 2", "label_text3": "Test Drawer 3", "label_text4": "Test Drawer 4"},
            "Sorting of the": {"Correspondence": "K1", "Line Items": "E8"},
            "Payment Advice Note Control": {
                "restricted to": True,
                "rows": "3",
                "None": False,
                "as many as req": False,
                "Do Not Consider Item Text": True
            },
            "Payment advice output according to no. of lines": {
                "Pymt adv. after ... lines": False,
                "Always Payt Adv": False,
                "NoPytAdv": True,
                "Pymt mthd valid to ... lines": True,
                "Distribute Items, .. lines per pymt": True
            }
        }
    }]
    await Maintenance_of_Company_Code_Data_for_a_Payment_Method_101044(targets=target_101044)

# # ── 17. Set Up Payment Methods for Each Country/Region for Payment Transactions (101972) ──
    target_101972 = [{
        "Country_Region": "IN",
        "Pymt_Meth": "9",
        "Description": "Bank Transfer 9",
        "Payment_Method_For": {"Outgoing payments": True, "Incoming payments": False},
        "Payment_Method_Classification": {"Bank transf": True},
        "Bank_Details": True,
        "IBAN_Required": True,
        "Document_Type_For_Payment": "ZP",
        "Clearing_Document_Type": "ZV",
        "Format": "IN_CGI_XML_CT"
    }]
    await Set_Up_Payment_Methods_for_Each_Country_Region_for_Payment_Transactions_101972(targets=target_101972)

    # # ── 18. Maintain Terms of Payment (102934) ──
    target_102934 = [{
        "Payment_Terms": "Z009",
        "Day_Limit": "0",
        "Sales_Text": "Test Sales Text Z001",
        "Own_Explanation": "Test Own Explanation",
        "Account_Type": {"Customer": True, "Supplier": False},
        "Baseline_Date_Calculation": {"Fixed_Day": "15", "Additional_Months": "1"},
        "Default_Baseline_Date": "Posting Date",
        "Pmnt_Block": {"Block_Key": "", "Block_Key_Checkbox": False},
        "Payment_Method": {"Method": "", "Method_Checkbox": False},
        "Installment_Payments": False,
        "Rec_Entries_Supplement": False,
        "Payment_Terms_Lines": [
            {"Percentage": "10", "No_of_Days": "14", "Fixed_Day": "5", "Additional_Months": "0"},
            {"Percentage": "5", "No_of_Days": "30", "Fixed_Day": "10", "Additional_Months": "1"},
            {"No_of_Days": "45", "Fixed_Day": "15", "Additional_Months": "2"}
        ],
        "Explanations": "Test Explanations block data.",
        "Hide_Entry_in_Input_Help": False
    }]
    await Maintain_Terms_of_Payment_102934(targets=target_102934)


if __name__ == "__main__":
    asyncio.run(main())
