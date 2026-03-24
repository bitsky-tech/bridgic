#!/bin/bash
# validate.sh — Validate a bridgic-amphibious project structure and imports.
#
# Usage: bash validate.sh [project_dir]
# Default project_dir: current directory

set -e

PROJECT_DIR="${1:-.}"
ERRORS=0

echo "=== Bridgic Amphibious Project Validator ==="
echo "Checking: $PROJECT_DIR"
echo ""

# 1. Check Python files exist
echo "[1/6] Checking project files..."
if [ ! -f "$PROJECT_DIR/agent.py" ] && [ ! -f "$PROJECT_DIR/main.py" ]; then
    echo "  WARNING: No agent.py or main.py found"
fi

PYTHON_FILES=$(find "$PROJECT_DIR" -name "*.py" -not -path "*/__pycache__/*" -not -path "*/.*" 2>/dev/null)
if [ -z "$PYTHON_FILES" ]; then
    echo "  ERROR: No Python files found in $PROJECT_DIR"
    ERRORS=$((ERRORS + 1))
else
    FILE_COUNT=$(echo "$PYTHON_FILES" | wc -l | tr -d ' ')
    echo "  Found $FILE_COUNT Python file(s)"
fi

# 2. Check that AmphibiousAutoma subclass exists with generic parameter
echo ""
echo "[2/6] Checking agent definition..."
AGENT_DEF=$(grep -r "class.*AmphibiousAutoma\[" $PYTHON_FILES 2>/dev/null || true)
if [ -z "$AGENT_DEF" ]; then
    echo "  ERROR: No AmphibiousAutoma subclass found with generic type parameter"
    echo "  Expected: class MyAgent(AmphibiousAutoma[MyContext]):"
    ERRORS=$((ERRORS + 1))
else
    echo "  OK: $AGENT_DEF"
fi

# 3. Check on_agent is defined
echo ""
echo "[3/6] Checking on_agent() method..."
ON_AGENT=$(grep -r "async def on_agent" $PYTHON_FILES 2>/dev/null || true)
if [ -z "$ON_AGENT" ]; then
    echo "  WARNING: No on_agent() method found (required unless workflow-only mode)"
else
    echo "  OK: on_agent() defined"
fi

# 4. Check tool functions are async
echo ""
echo "[4/6] Checking tool functions..."
SYNC_TOOLS=$(grep -n "^def.*-> str:" $PYTHON_FILES 2>/dev/null | grep -v "get_tools\|__" || true)
if [ -n "$SYNC_TOOLS" ]; then
    echo "  WARNING: Found potentially synchronous tool functions (must be async):"
    echo "$SYNC_TOOLS" | head -5
fi
ASYNC_TOOLS=$(grep -c "async def.*-> str:" $PYTHON_FILES 2>/dev/null | awk -F: '{sum += $2} END {print sum}')
echo "  Found $ASYNC_TOOLS async tool function(s)"

# 5. Check for common mistakes
echo ""
echo "[5/6] Checking for common mistakes..."

# Check for worker.arun(ctx) without keyword
BAD_ARUN=$(grep -n "\.arun([^=]*ctx\b" $PYTHON_FILES 2>/dev/null | grep -v "context=" || true)
if [ -n "$BAD_ARUN" ]; then
    echo "  ERROR: Found arun() call without keyword argument:"
    echo "  $BAD_ARUN"
    echo "  Fix: Use worker.arun(context=ctx)"
    ERRORS=$((ERRORS + 1))
fi

# Check for AmphibiousAutoma without generic parameter
BAD_GENERIC=$(grep -r "AmphibiousAutoma):" $PYTHON_FILES 2>/dev/null || true)
if [ -n "$BAD_GENERIC" ]; then
    echo "  ERROR: AmphibiousAutoma used without generic type parameter:"
    echo "  $BAD_GENERIC"
    echo "  Fix: class MyAgent(AmphibiousAutoma[MyContext]):"
    ERRORS=$((ERRORS + 1))
fi

# Check for sync tool functions used with FunctionToolSpec
SYNC_SPEC=$(grep -B5 "FunctionToolSpec.from_raw" $PYTHON_FILES 2>/dev/null | grep "^def " | grep -v "async def" || true)
if [ -n "$SYNC_SPEC" ]; then
    echo "  WARNING: Sync functions passed to FunctionToolSpec (should be async)"
fi

echo "  Done"

# 6. Check imports
echo ""
echo "[6/6] Checking imports..."
IMPORTS_OK=true
for module in "bridgic.amphibious"; do
    if grep -q "$module" $PYTHON_FILES 2>/dev/null; then
        echo "  OK: $module imported"
    fi
done

# Summary
echo ""
echo "=== Validation Complete ==="
if [ $ERRORS -gt 0 ]; then
    echo "FAILED: $ERRORS error(s) found"
    exit 1
else
    echo "PASSED: No errors found"
    exit 0
fi
