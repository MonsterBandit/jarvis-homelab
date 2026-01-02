#!/bin/bash
# IRR Stage 5 post-restore notification script
# Safe read-only audit after server powers on
# Compatible with NUT monitor and 'micro' workflow

# Wait a few seconds to ensure server fully booted
sleep 30

# Define paths
AUDIT_LOG="/opt/jarvis/brain-data/irr/irr_stage5_post_restore_audit.txt"
MARKER_FILE="/opt/jarvis/brain-data/irr/restore_pending"

# Header for audit
echo "### IRR Stage 5 – Post-Restore Verification (Read-Only Audit)" >> "$AUDIT_LOG"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')" >> "$AUDIT_LOG"

# Marker file check (silent if missing)
if [ -f "$MARKER_FILE" ]; then
    echo "Stage 5 Trigger: Marker present" >> "$AUDIT_LOG"
    ls -l "$MARKER_FILE" >> "$AUDIT_LOG"
else
    echo "Stage 5 Trigger: Marker absent" >> "$AUDIT_LOG"
fi

# ZFS Pool verification (read-only)
zpool status -v >> "$AUDIT_LOG" 2>&1

# Optional: include marker file listing silently
ls -l "$MARKER_FILE" 2>/dev/null >> "$AUDIT_LOG"

# Footer
echo "Stage 5 script completed — read-only audit appended." >> "$AUDIT_LOG"
echo "--------------------------------------------------------" >> "$AUDIT_LOG"
