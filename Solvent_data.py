import fitz  # PyMuPDF
import re
import pandas as pd
from pathlib import Path

# === FILE PATH ===
pdf_path = Path(r"C:\Users\raksh\OneDrive\Documents\Python\SolubilityML\ML_Solubility\CRC.Press.Handbook.of.Chemistry.and.Physics.85th.ed.eBook-LRN.pdf")

# === OPEN PDF ===
doc = fitz.open(pdf_path)
total_pages = len(doc)
print(f"\n✅ Loaded PDF successfully. Total pages: {total_pages}")

# === SET YOUR PAGE RANGE (adjust these if needed after verifying in the PDF) ===
# Example: section 15-16 to 15-25 is usually near page 2400–2450
start_page = 2423   # change if section starts elsewhere
end_page   = 2432   # change if section ends elsewhere

# === EXTRACT TEXT FROM THE RANGE ===
text_data = ""
for page_num in range(start_page, end_page):
    if page_num < total_pages:
        page = doc.load_page(page_num)
        text_data += f"\n--- PAGE {page_num+1} ---\n"
        text_data += page.get_text("text")

# Save raw extracted text
output_txt = pdf_path.parent / "CRC_solvent_constants_raw.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    f.write(text_data)
print(f"✅ Extracted raw text saved to: {output_txt}")

# === STEP 2: TRY PARSING LINES WITH COMMON SOLVENT TABLE FORMATS ===
pattern = re.compile(
    r"^([A-Za-z0-9\-\(\)\s]+)\s+([A-Z0-9\(\)\.]+)?\s+([-–]?\d+\.\d+|\d+)\s+([-–]?\d+\.\d+|\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",
    re.MULTILINE
)

rows = []
for match in pattern.finditer(text_data):
    name, formula, mp, bp, density, refr_index = match.groups()
    rows.append({
        "Solvent": name.strip(),
        "Formula": formula,
        "Melting_Point_C": float(mp),
        "Boiling_Point_C": float(bp),
        "Density_20C_gcm3": float(density),
        "Refractive_Index_nD20": float(refr_index)
    })

# === SAVE DATAFRAME ===
if rows:
    df = pd.DataFrame(rows)
    output_csv = pdf_path.parent / "CRC_solvent_properties_15_16_to_15_25.csv"
    df.to_csv(output_csv, index=False)
    print(f"✅ Parsed {len(df)} solvent entries and saved to:\n{output_csv}")
else:
    print("⚠️ No matches found. You may need to adjust the regex based on how text is spaced in your PDF.")
