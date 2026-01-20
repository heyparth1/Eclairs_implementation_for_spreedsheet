# ECLAIR Domain Context

## Column Descriptions & Codes

### Type/SNo
Specifies the sub-category of the Product Type.

**For PIPE:**
- **ALU**: Aluminium
- **COM**: Commercial
- **DAW**: Double Submerged Arc Welded
- **DSW**: Double Submerged Arc Welded
- **DUC**: Ductile
- **EFW**: Electric Fusion Welded
- **ERW**: Electric Resistance Welded
- **JCO**: JCO Formed
- **SAW**: Submerged Arc Welded
- **SML**: Seamless
- **SPR**: Spiral pipes (created from COILS)
- **SSM**: Stainless Seamless
- **STN**: Stainless
- **TEE**: Tee piece
- **UOE**: UOE Formed

**For BEAMS:**
- **BPB**: Bearing Pile Beam
- **IBM**: I-Beam
- **WFB**: Wide Flange Beam

**For SHEET:**
- **P**: Plate
- **S**: Sheet

**For COILS:**
- Every COIL has its own unique ID.

### Product ID Naming Conventions

**Common Identifier to identify each SKU.**

**For BEAMS:**
*   **Format:** `<OD><WALLT><Length>F<Type/SNo><Grade>`
*   **OD:** `XX.XXX` in Inches (Outer Diameter)
*   **WALLT:** `XX.XX` in Inches (Wall Thickness)
*   **Length:** `XX.XX` in Feet
*   **Example:** `06.00020.0003.00FWFB26` -> 6.000" OD, 20.00" Wall, 3.00' Length, WFB type, 26 Grade

**For PIPE:**
*   **Format:** `<OD><WALLT><Length>F<Type/SNo><Grade>`
*   **OD:** `00.XXX` in Inches (Outer Diameter)
*   **WALLT:** `X.XXX` in Inches (Wall Thickness)
*   **Length:** `XX.XX` in Feet
*   **Example:** `00.5000.14713.00FERW22` -> 0.500" OD, 0.147" Wall, 13.00' Length, ERW type, 22 Grade

**For COILS:**
*   **Format:** `<OD><WALLT><Type/SNo><Grade>`
*   **OD:** `XX.XXX` in Inches (Width)
*   **WALLT:** `X.XXX` in Inches (Thickness)
*   **Example:** `48.0000.180C9878 89` -> 48.000" Width, 0.180" Thickness, C9878 sub-category, 89 Grade

### Location (Yard/Warehouse Codes)
- **PPBC**, **SPIRAL**, **PPCAL**, **PPMTL** (Montreal), **DYMTOR**, **PPEDM**, **ECLIPS**, **NISKU2**, **PPCNTL**, **TROIS**, **PPEAST**, **POTTS**, **CANPHX**, **TYT_CSL**, **SPIRALCO_PLANT_1**, **RELWEL**, **TRANSCNTL**, **SEVEN_HORSES**, **FSD**, **LTEAST**, **STEWART**

### Measurement Definitions (CRITICAL)
- **OD (Outer Diameter)**: 
    - For **Pipes/Beams**: Outer Diameter in inches (Column: `OD`).
    - For **Coils**: Represents **WIDTH** in inches (Column: `OD`).
- **WALLT (Wall Thickness)**: Thickness in inches (Column: `WALLT`).
- **Length**: 
    - **NOT A COLUMN**. It is embedded in `Product ID`.
    - **Parsing Rule (Beams/Pipes)**: Characters 12-16 (Indices 11:16) of `Product ID`.
      - Format: `OD(6 chars) WALL(5 chars) LENGTH(5 chars) ...`
      - Example: `06.00020.00003.00...` -> Length is `03.00`.
      - Code: `pd.to_numeric(df['Product ID'].str[11:16], errors='coerce')`
    - **Coils DO NOT have Length** (measured by weight).
- **Grade**: Product grade (e.g., 22, 26, 89).

**NOTE**: We **DO NOT** use terms like "Schedule", "Height", or "class". Use only OD, Wall Thickness, Length, or Grade.
- **Current Stock**: Total quantity (Feet for pipes/beams, Pounds for coils)
- **Current Stock(Pcs)**: Number of pieces (Coils always 1)
- **Wt/Ft (lbs)**: Weight per foot (Pipes/Beams)
- **WT/Pce (lbs)**: Weight per piece. For coils = Current Stock (lbs).
- **Total Wt (Tons )**: ((Current Stock * Wt/Ft) / 2000)

### Origin Country
- **Product Description**: Contains the **Origin Country** of the product (e.g., "Korea", "USA", "Canada", "China").
- **Location**: Refers ONLY to the **Yard/Warehouse** where stock is held (e.g., PPBC, PPMTL). Does NOT refer to country of origin.

