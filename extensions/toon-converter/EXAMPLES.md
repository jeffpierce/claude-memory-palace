# TOON Converter Usage Examples

## CLI Usage

### Single File Conversion

Convert a JSONL file to TOON format with automatic output naming:
```bash
python -m extensions.toon-converter input.jsonl
# Creates input.toon in the same directory
```

Convert with explicit output path:
```bash
python -m extensions.toon-converter input.jsonl output.toon
```

Convert with output directory:
```bash
python -m extensions.toon-converter input.jsonl --output-dir ./converted/
# Creates converted/input.toon
```

### Batch Directory Conversion

Convert all JSONL files in a directory:
```bash
python -m extensions.toon-converter input_dir/ output_dir/
```

Convert with custom file pattern:
```bash
python -m extensions.toon-converter logs/ converted/ --pattern "transcript_*.jsonl"
```

### Conversion Modes

Maximum compression (default):
```bash
python -m extensions.toon-converter input.jsonl --mode aggressive
```

More readable output:
```bash
python -m extensions.toon-converter input.jsonl --mode conservative
```

## Expected Output

### Aggressive Mode
```
Conversion complete (aggressive mode)
============================================================
Input:             transcript.jsonl
Output:            transcript.toon
Original size:     45.2 KB
Converted size:    2.1 KB
Compression ratio: 95.4%
Estimated tokens:  ~525
============================================================
Records processed: 87
Records skipped:   3
```

### Conservative Mode
```
Conversion complete (conservative mode)
============================================================
Input:             transcript.jsonl
Output:            transcript.toon
Original size:     45.2 KB
Converted size:    3.8 KB
Compression ratio: 91.6%
Estimated tokens:  ~950
============================================================
Records processed: 87
Records skipped:   3
```

### Batch Conversion
```
Batch conversion complete (aggressive mode)
============================================================
Input directory:   transcripts/
Output directory:  converted/
Files processed:   5
Files failed:      0
Total original:    156.8 KB
Total converted:   7.2 KB
Avg compression:   95.4%
============================================================

Per-file results:
  [OK] session1.jsonl: 95.2% compression, 1.4 KB
  [OK] session2.jsonl: 96.1% compression, 890 B
  [OK] session3.jsonl: 94.8% compression, 2.1 KB
  [OK] session4.jsonl: 95.5% compression, 1.6 KB
  [OK] session5.jsonl: 95.4% compression, 1.2 KB
```

## MCP Server Usage

Start the MCP server:
```bash
python extensions/toon-converter/server.py
```

The server exposes one tool:

### convert_jsonl_to_toon

**Parameters:**
- `input_path` (required): Path to source JSONL file
- `output_path` (optional): Path for output file
- `mode` (optional): "conservative" or "aggressive" (default: "aggressive")

**Example Response:**
```json
{
  "success": true,
  "input_file": "/path/to/transcript.jsonl",
  "output_file": "/path/to/transcript.toon",
  "original_size": 45234,
  "converted_size": 2087,
  "original_size_formatted": "44.2 KB",
  "converted_size_formatted": "2.0 KB",
  "compression_ratio": 95.4,
  "estimated_tokens": 521,
  "records_processed": 87,
  "records_skipped": 3,
  "errors": 0
}
```

## Format Comparison

### Original JSONL (863 bytes)
```json
{"type": "user", "timestamp": "2026-02-10T12:00:00Z", "message": {"role": "user", "content": "Hello, can you help me?"}}
{"type": "assistant", "timestamp": "2026-02-10T12:00:05Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "Of course! I'd be happy to help. What do you need assistance with?"}]}}
{"type": "user", "timestamp": "2026-02-10T12:00:30Z", "message": {"role": "user", "content": "I need to convert some files to TOON format."}}
{"type": "assistant", "timestamp": "2026-02-10T12:00:35Z", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "The user wants to convert files to TOON format. I should explain what TOON is and how to use the converter."}, {"type": "text", "text": "TOON (Token-Optimized Notation) is a compact format. You can use the toon-converter extension to convert JSONL files."}]}}
```

### Aggressive Mode (268 bytes, 68.9% compression)
```
U: Hello, can you help me?
A: Of course! I'd be happy to help. What do you need assistance with?
U: I need to convert some files to TOON format.
A: TOON (Token-Optimized Notation) is a compact format. You can use the toon-converter extension to convert JSONL files.
```

### Conservative Mode (461 bytes, 46.6% compression)
```
[12:00:00] USER: Hello, can you help me?

[12:00:05] ASSISTANT: Of course! I'd be happy to help. What do you need assistance with?

[12:00:30] USER: I need to convert some files to TOON format.

[12:00:35] ASSISTANT: [THINKING: The user wants to convert files to TOON format. I should explain what TOON is and how to use the converter.]
TOON (Token-Optimized Notation) is a compact format. You can use the toon-converter extension to convert JSONL files.
```

## Error Handling

### File Not Found
```bash
$ python -m extensions.toon-converter nonexistent.jsonl
Error: Input path does not exist: nonexistent.jsonl
```

### Invalid Mode
```bash
$ python -m extensions.toon-converter input.jsonl --mode invalid
Error: Invalid mode 'invalid'. Must be 'conservative' or 'aggressive'.
```

### Missing Output for Batch
```bash
$ python -m extensions.toon-converter input_dir/
Error: Output directory required for batch conversion
Use: python -m extensions.toon-converter.cli input_dir/ output_dir/
```

## Integration Examples

### Memory Palace Workflow
```bash
# 1. Export Claude transcript to JSONL
# (using your preferred method)

# 2. Convert to TOON for analysis
python -m extensions.toon-converter session_transcript.jsonl

# 3. Use with memory_reflect tool
# The TOON file is now ready for reflection analysis
```

### Batch Processing Pipeline
```bash
# Process all transcripts from a month
python -m extensions.toon-converter \
    ~/claude_transcripts/2026-02/ \
    ~/toon_archives/2026-02/ \
    --mode aggressive

# Result: All JSONL files converted with ~95% compression
```
