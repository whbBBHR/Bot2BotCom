# Document Import - Quick Reference

## Installation
```bash
pip install pypdf python-docx markdown
```

## Basic Usage
```bash
python API_2_Api_com.py
→ Choose [4] Import documents
→ Enter file paths
→ Ask questions!
```

## Supported Formats
| Format | Extensions |
|--------|-----------|
| PDF | `.pdf` |
| Word | `.docx`, `.doc` |
| Text | `.txt` |
| Markdown | `.md`, `.markdown` |

## Main Features
- ✅ Multi-document import
- ✅ Document Q&A mode
- ✅ Persistent context
- ✅ Both Claude & GPT-4o
- ✅ Error handling
- ✅ Status indicator

## Example Workflow
```
[4] → Enter: report.pdf
    → Summary: 1 file, 4,523 words
    → Q&A: "What are the main findings?"
    → AI response based on document
```

## Privacy
⚠️ Documents sent to Claude & OpenAI APIs
✓ Files stay local (only text extracted)
❌ Don't upload sensitive documents

## Documentation
- Full guide: [DOCUMENT_IMPORT_GUIDE.md](DOCUMENT_IMPORT_GUIDE.md)
- Summary: [DOCUMENT_IMPORT_SUMMARY.md](DOCUMENT_IMPORT_SUMMARY.md)
- Source: [document_processor.py](document_processor.py)
