"""
SAP S/4HANA Procure-to-Pay (P2P) Configuration — MCP Tool Server
=================================================================

This MCP server exposes SAP S/4HANA P2P configuration automation tools
so that Claude can automatically select and invoke the correct tool based
on natural language user requests.

Each tool wraps a Playwright-based automation flow that:
  1. Opens a browser to the SAP Fiori Launchpad
  2. Logs in automatically
  3. Navigates to the correct SSCUI configuration screen
  4. Makes the requested changes (checkboxes, text fields, radio buttons, etc.)
  5. Saves and logs out

⚠️  IMPORTANT: Each tool call launches a real browser and takes 30-120 seconds.
"""

import asyncio
import json
import traceback
from typing import Optional
from mcp.server.fastmcp import FastMCP

import sys
import os

# Add parent directory to sys.path to allow importing P2P_configuration
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# ─── Import all P2P automation functions ───
from P2P_configuration import (
    activate_purchase_requisition_workflow_102911,
    configure_purchase_requisition_102888,
    activate_po_flexible_workflow_101097,
    configure_po_workflow_restart_conditions_103345,
    supply_invoices_101098,
    Document_Types_Contract_Change_101247,
    Create_New_Material_Group_102665,
    Edit_Existing_Material_Group_102665,
    Create_New_Picking_Locations_102130,
    Edit_Existing_Picking_Locations_102130,
    maintain_purchasing_organization_105939,
    Assign_Shipping_Points_102126,
    create_purchase_groups_102914,
    Entry_Aids_for_Items_Without_a_Material_Master_101602,
    Set_Tolerance_limits_101947,
    edit_storage_location_105933,
)

# ─── Initialize MCP Server ───
mcp = FastMCP(
    name="sap-p2p-config",

    instructions="""
You are connected to an SAP S/4HANA Procure-to-Pay (P2P) Configuration Automation Server.
PURPOSE
This server automates SAP S/4HANA IMG (Implementation Guide) configuration
activities across the Procure-to-Pay process. Each tool launches a real
Chromium browser via Playwright, logs into SAP Fiori Launchpad, navigates
to the correct SSCUI configuration screen, makes changes (checkboxes,
text fields, comboboxes, radio buttons), saves, and logs out.
TOOL SELECTION GUIDE — HOW TO PICK THE RIGHT TOOL
  1. USER MENTIONS A SSCUI NUMBER → Use the tool for that exact SSCUI.
     Example: "Configure SSCUI 102911" → activate_purchase_requisition_workflow

  2. USER MENTIONS WORKFLOW FOR A DOCUMENT TYPE:
     • "Purchase Requisition" + workflow   → activate_purchase_requisition_workflow (102911)
     • "Purchase Requisition" + processing → configure_purchase_requisition (102888)
     • "Purchase Order" + workflow          → activate_po_flexible_workflow (101097)
     • "Purchase Order" + restart           → configure_po_workflow_restart_conditions (103345)
     • "Supplier Invoice" + workflow        → activate_supplier_invoice_workflow (101098)
     • "Contract" + workflow                → activate_contract_change_workflow (101247)

  3. USER MENTIONS CREATE vs EDIT:
     • "Create/add/new material group"      → create_material_groups
     • "Edit/update/change material group"  → edit_material_groups
     • "Create/add/new picking location"    → create_picking_locations
     • "Edit/update/change picking location"→ edit_picking_locations
     ⚠️ ALWAYS ask the user if they want to CREATE NEW or EDIT EXISTING
        when the intent is ambiguous.

  4. USER MENTIONS TOLERANCE / LIMITS:
     → set_tolerance_limits (101947)

  5. USER MENTIONS SHIPPING POINTS:
     → assign_shipping_points (102126)

  6. USER MENTIONS PURCHASING GROUP:
     → create_purchasing_groups (102914)

  7. USER MENTIONS PURCHASING ORGANIZATION:
     → maintain_purchasing_organization (105939)
org_structure_tools/venv
  8. USER MENTIONS STORAGE LOCATION (org structure):
     → edit_storage_location (CBC API — fast, no browser)

  9. USER MENTIONS ENTRY AIDS / ITEMS WITHOUT MATERIAL:
     → configure_entry_aids_without_material_master (101602)

BEHAVIOR GUIDELINES
  • CONFIRM BEFORE EXECUTING: Each tool launches a real browser and
    modifies live SAP configuration. Always confirm the exact parameters
    with the user before calling a tool.

  • EXECUTION TIME: Browser-based tools take 30-120 seconds each.
    Warn the user that the operation will take time.
    The edit_storage_location tool is the exception — it uses a REST API
    and completes in 5-10 seconds.

  • LOCK HANDLING: SAP configuration tables can be locked by other users.
    If a tool fails due to "Locked Data", inform the user and suggest
    retrying later. The tools auto-detect and abort on locks.

  • SAVE BEHAVIOR: All tools auto-save after making changes. There is
    a customizing transport request confirmation dialog that is handled
    automatically.

  • DATA VALIDATION: The tools do NOT validate whether IDs exist before
    attempting changes. If the user provides a wrong ID, the tool will
    fail gracefully and return an error message.

  • BULK OPERATIONS: Most tools accept arrays of targets. When the user
    has multiple items, batch them into a single call rather than making
    multiple separate calls.

  • BE CONSERVATIVE: Never guess field values. If the user hasn't specified
    a required field, ask for it. Never auto-fill with defaults.

  • DOWNSTREAM IMPACT: When configuring workflows or tolerance limits,
    explain to the user how changes may affect downstream MM/FI processes
    (e.g., enabling a workflow restart condition means POs will re-enter
    approval when that attribute changes).
GOAL
Help SAP consultants and administrators safely configure, validate,
and automate SAP S/4HANA Procure-to-Pay IMG settings with confidence.
Always prioritize accuracy, confirmation, and user safety over speed.
"""
)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 1: SSCUI 102911 — Purchase Requisition Workflow
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="activate_purchase_requisition_workflow",
    description="""Activates or deactivates the Purchase Requisition Flexible Workflow settings in SAP S/4HANA (SSCUI 102911).
PURPOSE:
    Toggles two checkboxes per Purchase Requisition document type:
      • OvRel (Overall Release) — column 3
      • Sce (Scenario-based workflow) — column 4
WHEN TO USE:
    Use this tool when the user asks to:
      - "Enable/disable overall release for purchase requisitions"
      - "Activate scenario-based workflow for PR document type NB"
      - "Configure purchase requisition workflow settings"
      - "Set up PR flexible workflow for document types"
      - Any mention of SSCUI 102911
INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "target_name" (str): The document type code visible in the SAP table (e.g., "NB", "NBS", "RV")
      - "opt1_val" (bool): True to CHECK the OvRel checkbox, False to UNCHECK it
      - "opt2_val" (bool): True to CHECK the Sce checkbox, False to UNCHECK it

EXAMPLE INPUT:
    [
        {"target_name": "NB",  "opt1_val": true,  "opt2_val": true},
        {"target_name": "NBS", "opt1_val": false, "opt2_val": true},
        {"target_name": "RV",  "opt1_val": true,  "opt2_val": false}
    ]
IMPORTANT NOTES:
    - Values must be boolean (true/false), NOT strings
    - "target_name" must exactly match the code shown in the SAP table
    - This tool launches a browser and takes 30-90 seconds to complete
    - Multiple document types can be configured in a single call
"""
)
async def activate_purchase_requisition_workflow(targets: list[dict]) -> str:
    """Activate/deactivate Purchase Requisition Flexible Workflow (SSCUI 102911)."""
    try:
        await activate_purchase_requisition_workflow_102911(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Purchase Requisition Workflow updated for {len(targets)} document type(s): {[t['target_name'] for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to update PR workflow: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 2: SSCUI 102888 — Configure Purchase Requisition
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="configure_purchase_requisition",
    description="""Configures Purchase Requisition processing settings in SAP S/4HANA (SSCUI 102888).

PURPOSE:
    For each PR attribute row, sets two checkbox flags:
      • SS Proc (Source of Supply Processing) — column 3
      • Prf Proc (Preferred Processing) — column 4

WHEN TO USE:
    Use this tool when the user asks to:
      - "Configure purchase requisition processing"
      - "Enable source of supply for PR attributes"
      - "Set preferred processing for price unit in purchase requisitions"
      - "Manage PR workflow restart conditions"
      - Any mention of SSCUI 102888

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "target_name" (str): The attribute name in the SAP table (e.g., "Price Unit", "Currency Key")
      - "ss_proc" (bool): True to CHECK Source of Supply checkbox, False to UNCHECK
      - "prf_proc" (bool): True to CHECK Preferred Processing checkbox, False to UNCHECK

EXAMPLE INPUT:
    [
        {"target_name": "Price Unit",                 "ss_proc": true,  "prf_proc": false},
        {"target_name": "Purchase Requisition Price",  "ss_proc": true,  "prf_proc": true},
        {"target_name": "Currency Key",               "ss_proc": true,  "prf_proc": false},
        {"target_name": "Quantity of Items",           "ss_proc": true,  "prf_proc": true}
    ]

IMPORTANT NOTES:
    - Values must be boolean (true/false), NOT strings
    - "target_name" must exactly match the attribute text in the SAP table
    - This tool launches a browser and takes 30-90 seconds to complete
"""
)
async def configure_purchase_requisition(targets: list[dict]) -> str:
    """Configure Purchase Requisition processing (SSCUI 102888)."""
    try:
        await configure_purchase_requisition_102888(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Purchase Requisition configured for {len(targets)} attribute(s): {[t['target_name'] for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to configure PR: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 3: SSCUI 101097 — Activate PO Flexible Workflow
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="activate_po_flexible_workflow",
    description="""Activates or deactivates the Purchase Order (PO) Flexible Workflow per document type in SAP S/4HANA (SSCUI 101097).

PURPOSE:
    Toggles the "Scenario-based Workflow" checkbox for each PO document type.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Enable/disable PO flexible workflow"
      - "Activate scenario-based workflow for purchase order type NB"
      - "Configure purchase order document type workflow"
      - "Turn on/off flexible workflow for PO types"
      - Any mention of SSCUI 101097

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Type" (str): The PO document type code (e.g., "NB", "NB2", "NBAI", "NBIC")
      - "Scenario_based_workflow" (bool): True to CHECK, False to UNCHECK

EXAMPLE INPUT:
    [
        {"Type": "NB",   "Scenario_based_workflow": false},
        {"Type": "NB2",  "Scenario_based_workflow": true},
        {"Type": "NBAI", "Scenario_based_workflow": true},
        {"Type": "NBIC", "Scenario_based_workflow": true}
    ]
IMPORTANT NOTES:
    - Values must be boolean (true/false)
    - "Type" must match the code in the SAP table exactly
"""
)
async def activate_po_flexible_workflow(targets: list[dict]) -> str:
    """Activate PO Flexible Workflow per document type (SSCUI 101097)."""
    try:
        await activate_po_flexible_workflow_101097(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"PO Flexible Workflow updated for {len(targets)} type(s): {[t['Type'] for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to update PO workflow: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 4: SSCUI 103345 — PO Workflow Restart Conditions
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="configure_po_workflow_restart_conditions",
    description="""Configures the conditions that trigger a restart of the Purchase Order Flexible Workflow in SAP S/4HANA (SSCUI 103345).

PURPOSE:
    For each PO attribute, sets:
      • Enable (checkbox) — whether changes to this attribute trigger a workflow restart
      • Restart Type (combobox) — "Always Restart" or "Conditional Restart"

WHEN TO USE:
    Use this tool when the user asks to:
      - "Configure PO workflow restart conditions"
      - "Enable restart when company code changes on a PO"
      - "Set conditional restart for purchasing organization"
      - "Manage which PO attribute changes trigger workflow restart"
      - Any mention of SSCUI 103345

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Purchase_Order_Attributes" (str): The attribute name (e.g., "Company Code", "Plant", "Currency")
      - "Enable" (bool): True to enable restart for this attribute, False to disable
      - "Restart_Type" (str): "Always Restart" or "Conditional Restart" (ignored if Enable is False, use empty string "")

EXAMPLE INPUT:
    [
        {"Purchase_Order_Attributes": "Company Code",           "Enable": true,  "Restart_Type": "Always Restart"},
        {"Purchase_Order_Attributes": "Purchasing Group",       "Enable": false, "Restart_Type": ""},
        {"Purchase_Order_Attributes": "Purchasing Organization","Enable": true,  "Restart_Type": "Conditional Restart"},
        {"Purchase_Order_Attributes": "Total Net Order Value",  "Enable": true,  "Restart_Type": "Conditional Restart"},
        {"Purchase_Order_Attributes": "Incoterms",             "Enable": false, "Restart_Type": ""},
        {"Purchase_Order_Attributes": "Outline Agreement",     "Enable": true,  "Restart_Type": "Always Restart"},
        {"Purchase_Order_Attributes": "Material Group",        "Enable": true,  "Restart_Type": "Always Restart"},
        {"Purchase_Order_Attributes": "Currency",              "Enable": false, "Restart_Type": ""},
        {"Purchase_Order_Attributes": "Plant",                 "Enable": true,  "Restart_Type": "Always Restart"}
    ]

IMPORTANT NOTES:
    - "Enable" must be boolean (true/false)
    - "Restart_Type" is a string, use "" (empty) when Enable is false
    - Valid Restart_Type values: "Always Restart", "Conditional Restart"
    - "Purchase_Order_Attributes" must match the text in SAP exactly
"""
)
async def configure_po_workflow_restart_conditions(targets: list[dict]) -> str:
    """Configure PO Workflow Restart Conditions (SSCUI 103345)."""
    try:
        await configure_po_workflow_restart_conditions_103345(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"PO Restart Conditions updated for {len(targets)} attribute(s): {[t['Purchase_Order_Attributes'] for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to configure PO restart: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 5: SSCUI 101098 — Supplier Invoice Workflow
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="activate_supplier_invoice_workflow",
    description="""Activates or deactivates the Flexible Workflow for Supplier Invoices in SAP S/4HANA (SSCUI 101098).

PURPOSE:
    Toggles three global checkboxes on the supplier invoice workflow screen:
      1. "Payment Block: Flexible Workflow is Active"
      2. "Release Completed Invoice: Flexible Workflow is Active"
      3. "Check Authorizations for Flexible Workflow Steps"

WHEN TO USE:
    Use this tool when the user asks to:
      - "Enable/disable supplier invoice workflow"
      - "Activate payment block workflow for invoices"
      - "Configure supplier invoice flexible workflow"
      - "Turn on authorization checks for invoice workflow"
      - Any mention of SSCUI 101098

INPUT FORMAT:
    Three separate boolean parameters (NOT a list):
      - "payment_block" (bool): True to ENABLE payment block workflow, False to DISABLE
      - "release_completed" (bool): True to ENABLE release completed invoice workflow, False to DISABLE
      - "check_auth" (bool): True to ENABLE authorization checks, False to DISABLE

EXAMPLE INPUT:
    {
        "payment_block": false,
        "release_completed": true,
        "check_auth": false
    }

IMPORTANT NOTES:
    - This tool takes 3 individual boolean parameters, NOT a list of targets
    - All three parameters are required
    - Unlike other P2P tools, this screen has only 3 fixed checkboxes (not a data table)
"""
)
async def activate_supplier_invoice_workflow(
    payment_block: bool,
    release_completed: bool,
    check_auth: bool
) -> str:
    """Activate Flexible Workflow for Supplier Invoices (SSCUI 101098)."""
    try:
        await supply_invoices_101098(
            payment_block=payment_block,
            release_completed=release_completed,
            check_auth=check_auth
        )
        return json.dumps({
            "status": "success",
            "message": f"Supplier Invoice Workflow updated: payment_block={payment_block}, release_completed={release_completed}, check_auth={check_auth}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to update supplier invoice workflow: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 6: SSCUI 101247 — Document Types Contract Change
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="activate_contract_change_workflow",
    description="""Activates or deactivates the Flexible Workflow for Contract Change document types in SAP S/4HANA (SSCUI 101247).

PURPOSE:
    Toggles the "Scenario-based Workflow" checkbox for each contract document type.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Enable contract change workflow"
      - "Activate scenario-based workflow for contract type MK"
      - "Configure flexible workflow for contract document types"
      - "Turn on/off workflow for contract changes"
      - Any mention of SSCUI 101247

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "type_code" (str): The contract document type code (e.g., "MK", "CWK")
      - "Scenario_based_workflow" (bool): True to CHECK, False to UNCHECK

EXAMPLE INPUT:
    [
        {"type_code": "MK",  "Scenario_based_workflow": true},
        {"type_code": "CWK", "Scenario_based_workflow": true}
    ]

IMPORTANT NOTES:
    - Values must be boolean (true/false)
    - "type_code" must match the code in the SAP table exactly
"""
)
async def activate_contract_change_workflow(targets: list[dict]) -> str:
    """Activate Contract Change Flexible Workflow (SSCUI 101247)."""
    try:
        await Document_Types_Contract_Change_101247(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Contract Change Workflow updated for {len(targets)} type(s): {[t['type_code'] for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to update contract workflow: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 7: SSCUI 102665 — Create Material Groups
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="create_material_groups",
    description="""Creates NEW Material Group entries in SAP S/4HANA (SSCUI 102665).

PURPOSE:
    Adds brand new rows to the Material Groups configuration table. This tool clicks
    "New Entries" and fills in all fields for each new material group.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Create a new material group"
      - "Add material group W1201 for Water"
      - "Define new material groups in SAP"
      - "Set up new material group entries"
      - Any mention of creating/adding material groups or SSCUI 102665 with NEW entries

    ⚠️ Do NOT use this tool to EDIT existing entries — use "edit_material_groups" instead.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Matl_Group" (str): Material group ID code (e.g., "W1201") — REQUIRED, must be unique
      - "Material_Group_Desc" (str): Description in title case (e.g., "Water")
      - "AGrp" (str): Authorization group code (e.g., "1538")
      - "DUW" (str): Default unit of weight (e.g., "BTU", "D")
      - "Description_2" (str): Secondary description (e.g., "Water for drinking")

EXAMPLE INPUT:
    [
        {
            "Matl_Group": "W1201",
            "Material_Group_Desc": "Water",
            "AGrp": "1538",
            "DUW": "BTU",
            "Description_2": "Water for drinking"
        },
        {
            "Matl_Group": "W1202",
            "Material_Group_Desc": "Water Vapor",
            "AGrp": "9538",
            "DUW": "D",
            "Description_2": "Water vapor"
        }
    ]

IMPORTANT NOTES:
    - "Matl_Group" ID must not already exist in SAP — use "edit_material_groups" for existing entries
    - Multiple entries can be created in a single call (bulk creation)
    - Fields are automatically formatted (IDs → uppercase, descriptions → title case)
"""
)
async def create_material_groups(targets: list[dict]) -> str:
    """Create new Material Group entries (SSCUI 102665)."""
    try:
        await Create_New_Material_Group_102665(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Created {len(targets)} material group(s): {[t.get('Matl_Group') for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to create material groups: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 8: SSCUI 102665 — Edit Existing Material Groups
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="edit_material_groups",
    description="""Edits EXISTING Material Group entries in SAP S/4HANA (SSCUI 102665).

PURPOSE:
    Finds existing rows by their Material Group ID and updates their description,
    authorization group, default unit of weight, or secondary description.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Edit/update material group P000"
      - "Change the description of material group W1201"
      - "Modify an existing material group"
      - "Update material group authorization group"
      - Any mention of editing/updating/modifying material groups or SSCUI 102665 with EXISTING entries

    ⚠️ Do NOT use this tool to CREATE new entries — use "create_material_groups" instead.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Matl_Group" (str): Material group ID to find and edit — REQUIRED (this is the lookup key)
      - "Material_Group_Desc" (str, optional): New description
      - "AGrp" (str, optional): New authorization group code
      - "DUW" (str, optional): New default unit of weight
      - "Description_2" (str, optional): New secondary description

    Only include fields you want to change. Omit fields to leave them unchanged.

EXAMPLE INPUT:
    [
        {
            "Matl_Group": "P000",
            "Material_Group_Desc": "Contract Type",
            "AGrp": "5437",
            "DUW": "D",
            "Description_2": "Contract Type group"
        }
    ]

IMPORTANT NOTES:
    - "Matl_Group" must already exist in the SAP table
    - Only include fields you want to change — omitted fields remain unchanged
    - This tool searches for the row by ID, scrolling through the table if needed
"""
)
async def edit_material_groups(targets: list[dict]) -> str:
    """Edit existing Material Group entries (SSCUI 102665)."""
    try:
        await Edit_Existing_Material_Group_102665(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Edited {len(targets)} material group(s): {[t.get('Matl_Group') for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to edit material groups: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 9: SSCUI 102130 — Create Picking Locations
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="create_picking_locations",
    description="""Creates NEW Picking Location assignments in SAP S/4HANA (SSCUI 102130).

PURPOSE:
    Assigns a Storage Location to a combination of Shipping Point + Plant + Shipping Condition.
    This tool clicks "New Entries" and fills in all fields for each new assignment.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Create a new picking location"
      - "Assign picking location for shipping point 1003"
      - "Add storage location assignment for plant 1003"
      - "Set up new picking location assignments"
      - Any mention of creating/adding picking locations or SSCUI 102130 with NEW assignments

    ⚠️ Do NOT use this tool to EDIT existing assignments — use "edit_picking_locations" instead.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ShPt" (str): Shipping Point code (e.g., "1003")
      - "Plnt" (str): Plant code (e.g., "1003")
      - "SC" (str): Shipping Condition code (e.g., "10", "20")
      - "Stor" (str): Storage Location code to assign (e.g., "FG01", "181R")

EXAMPLE INPUT:
    [
        {"ShPt": "1003", "Plnt": "1003", "SC": "10", "Stor": "FG01"},
        {"ShPt": "9001", "Plnt": "9001", "SC": "20", "Stor": "181R"}
    ]

IMPORTANT NOTES:
    - All four fields are required for each entry
    - The combination ShPt+Plnt+SC must not already exist — use "edit_picking_locations" for existing rows
    - Values are automatically converted to uppercase
    - Multiple assignments can be created in a single call
"""
)
async def create_picking_locations(targets: list[dict]) -> str:
    """Create new Picking Location assignments (SSCUI 102130)."""
    try:
        await Create_New_Picking_Locations_102130(targets=targets)
        keys = [f"{t.get('ShPt')}/{t.get('Plnt')}/{t.get('SC')}" for t in targets]
        return json.dumps({
            "status": "success",
            "message": f"Created {len(targets)} picking location(s): {keys}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to create picking locations: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 10: SSCUI 102130 — Edit Existing Picking Locations
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="edit_picking_locations",
    description="""Edits EXISTING Picking Location assignments in SAP S/4HANA (SSCUI 102130).

PURPOSE:
    Finds existing rows by their Shipping Point + Plant + Shipping Condition combination,
    then updates the Storage Location assignment.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Edit/change picking location for shipping point 1003"
      - "Update storage location for an existing picking assignment"
      - "Change the storage location assigned to plant 1003 SC 10"
      - "Modify existing picking location assignments"
      - Any mention of editing/updating picking locations or SSCUI 102130 with EXISTING rows

    ⚠️ Do NOT use this tool to CREATE new assignments — use "create_picking_locations" instead.

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "ShPt" (str): Shipping Point code (lookup key)
      - "Plnt" (str): Plant code (lookup key)
      - "SC" (str): Shipping Condition code (lookup key)
      - "Stor" (str): New Storage Location code to set

EXAMPLE INPUT:
    [
        {"ShPt": "1003", "Plnt": "1003", "SC": "10", "Stor": "RM01"}
    ]

IMPORTANT NOTES:
    - ShPt + Plnt + SC form the composite lookup key — the row must already exist
    - Only "Stor" (Storage Location) can be changed
    - The tool auto-detects column indices from the table header
"""
)
async def edit_picking_locations(targets: list[dict]) -> str:
    """Edit existing Picking Location assignments (SSCUI 102130)."""
    try:
        await Edit_Existing_Picking_Locations_102130(targets=targets)
        keys = [f"{t.get('ShPt')}/{t.get('Plnt')}/{t.get('SC')}" for t in targets]
        return json.dumps({
            "status": "success",
            "message": f"Edited {len(targets)} picking location(s): {keys}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to edit picking locations: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 11: SSCUI 105939 — Maintain Purchasing Organization
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="maintain_purchasing_organization",
    description="""Edits the description of existing Purchasing Organizations in SAP S/4HANA (SSCUI 105939).

PURPOSE:
    Finds a Purchasing Organization by its POrg code and updates its description field.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Edit purchasing organization 1001 description"
      - "Update the name of a purchasing organization"
      - "Maintain purchasing organization details"
      - "Change purchasing org description"
      - Any mention of SSCUI 105939 or editing purchasing organizations

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "POrg" (str): Purchasing Organization code to find (e.g., "1001")
      - "Description" (str): New description (max 20 characters, auto-truncated)

EXAMPLE INPUT:
    [
        {"POrg": "1001", "Description": "Central Purchasing"}
    ]

IMPORTANT NOTES:
    - Description is truncated to 20 characters automatically
    - POrg code must already exist in the SAP table
"""
)
async def maintain_purchasing_organization(targets: list[dict]) -> str:
    """Maintain Purchasing Organization details (SSCUI 105939)."""
    try:
        await maintain_purchasing_organization_105939(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Updated {len(targets)} purchasing org(s): {[t.get('POrg') for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to maintain purchasing org: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 12: SSCUI 102126 — Assign Shipping Points
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="assign_shipping_points",
    description="""Assigns or updates Shipping Point determination rules in SAP S/4HANA (SSCUI 102126).

PURPOSE:
    For each combination of Shipping Condition + Loading Group + Plant + Proposed Shipping Point,
    sets the alternative shipping points (up to 11 alternatives in subsequent columns).

WHEN TO USE:
    Use this tool when the user asks to:
      - "Assign shipping points for plant 1002"
      - "Configure shipping point determination"
      - "Set alternative shipping points"
      - "Update shipping point assignment rules"
      - Any mention of SSCUI 102126 or shipping point assignment/determination

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Shipping_Condition" (str): Shipping condition code (e.g., "01")
      - "Loading_Group" (str): Loading group code (e.g., "0001")
      - "Plant" (str): Plant code (e.g., "1002")
      - "Proposed_Shipping_Point" (str): Proposed shipping point code (e.g., "1002")
      - "Alternative_Shipping_Points" (list[str]): List of up to 11 alternative shipping point codes.
        Use "" (empty string) for positions where no alternative is needed.

EXAMPLE INPUT:
    [
        {
            "Shipping_Condition": "01",
            "Loading_Group": "0001",
            "Plant": "1002",
            "Proposed_Shipping_Point": "1002",
            "Alternative_Shipping_Points": ["1002", "1002", "1002", "", "1002", "", "1002", "1002", "1002", "", ""]
        }
    ]

IMPORTANT NOTES:
    - Shipping_Condition + Loading_Group + Plant + Proposed_Shipping_Point form the lookup key
    - Alternative_Shipping_Points is a list of strings — use "" for empty positions
    - The row must already exist in the table
"""
)
async def assign_shipping_points(targets: list[dict]) -> str:
    """Assign Shipping Points determination (SSCUI 102126)."""
    try:
        await Assign_Shipping_Points_102126(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Assigned shipping points for {len(targets)} rule(s)"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to assign shipping points: {str(e)}",
            "traceback": traceback.format_exc()
        })


# TOOL 13: SSCUI 102914 — Create/Edit Purchasing Groups
@mcp.tool(
    name="create_purchasing_groups",
    description="""Creates or edits Purchasing Group entries in SAP S/4HANA (SSCUI 102914).
PURPOSE:
    Finds an existing Purchasing Group by its code and updates its contact details:
    description, telephone, fax, email, and extension.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Create/edit purchasing group 600"
      - "Update purchasing group contact details"
      - "Set email for purchasing group"
      - "Configure purchasing group telephone and fax"
      - Any mention of SSCUI 102914 or purchasing groups

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Pur_Grp" (str): Purchasing Group code (e.g., "600") — REQUIRED (lookup key)
      - "Description" (str, optional): Description (max 18 chars)
      - "Tel.No. Pur.Grp" or "Tel_No_Pur_Grp" (str, optional): Group telephone (max 12 chars)
      - "Fax Number" or "Fax" (str, optional): Fax number (max 31 chars, auto-formatted as "XXX XXX XXXX")
      - "Telephone" (str, optional): Direct telephone (max 30 chars, auto-formatted as "XXX XXX XXXX")
      - "Extension" (str, optional): Extension number (max 10 chars)
      - "Email" (str, optional): Email address (max 132 chars)

EXAMPLE INPUT:
    [
        {
            "Pur_Grp": "600",
            "Description": "Assets",
            "Email": "p01@example.com",
            "Fax Number": "3456789",
            "Telephone": "6586987456",
            "Extension": "91",
            "Tel.No. Pur.Grp": "91"
        }
    ]
IMPORTANT NOTES:
    - "Pur_Grp" is the lookup key and must exist in the SAP table
    - Phone numbers are auto-formatted to "XXX XXX XXXX" pattern
    - Fields have character limits (auto-truncated): Description(18), Tel(12), Fax(31), Phone(30), Ext(10), Email(132)
    - Only include fields you want to change
"""
)
async def create_purchasing_groups(targets: list[dict]) -> str:
    """Create/edit Purchasing Groups (SSCUI 102914)."""
    try:
        await create_purchase_groups_102914(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Updated {len(targets)} purchasing group(s): {[t.get('Pur_Grp') for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to update purchasing groups: {str(e)}",
            "traceback": traceback.format_exc()
        })


# TOOL 14: SSCUI 101602 — Entry Aids Without Material Master
@mcp.tool(
    name="configure_entry_aids_without_material_master",
    description="""Configures Entry Aids for Items Without a Material Master in SAP S/4HANA (SSCUI 101602).

PURPOSE:
    Finds a Material Group row by its code and updates its description,
    valuation class (ValCl), and purchasing value key (PurValK).
WHEN TO USE:
    Use this tool when the user asks to:
      - "Configure entry aids for items without material master"
      - "Set valuation class for material group GRP001"
      - "Update purchasing value key for non-material items"
      - "Edit entry aids configuration"
      - Any mention of SSCUI 101602 or entry aids for items without a material master
INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "Mat_Grp" (str): Material Group code to find and edit — REQUIRED (lookup key)
      - "Mat_Grp_Descr" (str, optional): Material Group description
      - "ValCl" (str, optional): Valuation Class code (e.g., "3000")
      - "PurValK" (str, optional): Purchasing Value Key (e.g., "1")
EXAMPLE INPUT:
    [
        {
            "Mat_Grp": "GRP001",
            "Mat_Grp_Descr": "Raw Materials goods",
            "ValCl": "3000",
            "PurValK": "1"
        }
    ]
IMPORTANT NOTES:
    - "Mat_Grp" must already exist in the SAP table
    - Only include fields you want to change
    - Uses the same table interaction pattern as material groups
"""
)
async def configure_entry_aids_without_material_master(targets: list[dict]) -> str:
    """Configure Entry Aids for Items Without Material Master (SSCUI 101602)."""
    try:
        await Entry_Aids_for_Items_Without_a_Material_Master_101602(targets=targets)
        return json.dumps({
            "status": "success",
            "message": f"Configured entry aids for {len(targets)} material group(s): {[t.get('Mat_Grp') for t in targets]}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to configure entry aids: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 15: SSCUI 101947 — Set Tolerance Limits
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="set_tolerance_limits",
    description="""Sets Tolerance Limits for invoice verification in SAP S/4HANA (SSCUI 101947).

PURPOSE:
    Configures tolerance limits per Company Code and Tolerance Key. Each entry can have
    up to 4 subsections with radio buttons and value inputs:
      • Lower Limit → Absolute (radio: Check Limit / Do Not Check, value input)
      • Lower Limit → Percentage (radio: Check Limit / Do Not Check, value input)
      • Upper Limit → Absolute (radio: Check Limit / Do Not Check, value input)
      • Upper Limit → Percentage (radio: Check Limit / Do Not Check, value input)

WHEN TO USE:
    Use this tool when the user asks to:
      - "Set tolerance limits for company code 1010"
      - "Configure invoice tolerance VP for company 9000"
      - "Set lower limit absolute check to 10.00"
      - "Update tolerance percentages for invoice verification"
      - "Configure tolerance key BD/DW/VP/AN/AP/PP/ST/BR for company code XXXX"
      - Any mention of SSCUI 101947, tolerance limits, or invoice tolerance

INPUT FORMAT:
    A JSON list of targets. Each target has:
      - "company_code" (str): Company Code (e.g., "1010", "9000")
      - "tolerance_key" (str): Tolerance Key code (e.g., "VP", "BD", "DW", "AN", "AP", "PP")
      - "lower_limit" (dict, optional): Lower limit settings with sub-keys:
          - "absolute" (dict, optional): {"check": bool, "value": str}
          - "percentage" (dict, optional): {"check": bool, "value": str}
      - "upper_limit" (dict, optional): Upper limit settings with sub-keys:
          - "absolute" (dict, optional): {"check": bool, "value": str}
          - "percentage" (dict, optional): {"check": bool, "value": str}

    For each sub-section:
      - "check" (bool): True = select "Check Limit" radio, False = select "Do Not Check" radio
      - "value" (str): The numeric value to enter (e.g., "10.00", "5.00"). Use "" if no value needed.

EXAMPLE INPUT:
    [
        {
            "company_code": "1010",
            "tolerance_key": "VP",
            "lower_limit": {
                "absolute":   {"check": true,  "value": "10.00"},
                "percentage": {"check": true,  "value": "5.00"}
            },
            "upper_limit": {
                "absolute":   {"check": false, "value": ""},
                "percentage": {"check": true,  "value": "10.00"}
            }
        }
    ]

IMPORTANT NOTES:
    - This is the most complex P2P tool — it navigates to a detail screen for each target
    - "check" is boolean: true = "Check Limit", false = "Do Not Check"
    - "value" is always a string (e.g., "10.00") even for numeric amounts
    - Values are truncated to max 7 characters
    - Omit lower_limit/upper_limit sections that don't exist on the screen
    - Multiple company code + tolerance key combinations can be processed in one call
    - Each target requires navigating to a detail screen, saving, and returning — expect 60-120 seconds per target
"""
)
async def set_tolerance_limits(targets: list[dict]) -> str:
    """Set Tolerance Limits for invoice verification (SSCUI 101947)."""
    try:
        await Set_Tolerance_limits_101947(targets=targets)
        keys = [f"{t.get('company_code')}/{t.get('tolerance_key')}" for t in targets]
        return json.dumps({
            "status": "success",
            "message": f"Tolerance limits set for {len(targets)} target(s): {keys}"
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to set tolerance limits: {str(e)}",
            "traceback": traceback.format_exc()
        })


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL 16: Edit Storage Location (CBC API — no browser)
# ═══════════════════════════════════════════════════════════════════════════════

@mcp.tool(
    name="edit_storage_location",
    description="""Updates the name and city of a Storage Location via the SAP CBC API (no browser needed).

PURPOSE:
    Updates the Name and City fields of an existing Storage Location in the SAP
    Central Business Configuration (CBC) organizational structure.

WHEN TO USE:
    Use this tool when the user asks to:
      - "Edit storage location name"
      - "Update storage location city"
      - "Change storage location description"
      - "Rename storage location FG01"
      - Any request to modify storage location details in the org structure

    NOTE: This tool uses the CBC REST API, NOT Playwright. It is much faster than browser-based tools.

INPUT FORMAT:
    Three separate parameters:
      - "storageid" (str): Storage Location ID (e.g., "FG01", "RM01")
      - "city" (str): City name. If empty (""), the tool auto-inherits city from the parent org unit.
      - "description" (str): New name/description for the storage location (max 16 chars, auto-truncated)

EXAMPLE INPUT:
    {
        "storageid": "FG01",
        "city": "Hyderabad",
        "description": "Finished Goods"
    }

    Or with auto-inherited city:
    {
        "storageid": "FG01",
        "city": "",
        "description": "Finished Goods"
    }

IMPORTANT NOTES:
    - This tool uses the CBC API, NOT the browser — it's much faster (5-10 seconds)
    - If "city" is empty, the tool automatically finds the city from the parent org unit hierarchy
    - "description" is truncated to 16 characters automatically
    - The storage location must already exist in the org structure
"""
)
async def edit_storage_location(
    storageid: str,
    city: str = "",
    description: str = ""
) -> str:
    """Edit Storage Location name/city via CBC API."""
    try:
        result = await edit_storage_location_105933(
            storageid=storageid,
            city=city,
            description=description
        )
        return json.dumps({
            "status": "success",
            "message": f"Storage Location '{storageid}' updated. Description: '{description}', City: '{city or 'auto-inherited'}'",
            "api_response": str(result)
        })
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to edit storage location: {str(e)}",
            "traceback": traceback.format_exc()
        })

if __name__ == "__main__":
    mcp.run()
