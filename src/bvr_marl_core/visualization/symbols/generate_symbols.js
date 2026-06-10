/**
 * Generate MIL-STD-2525C compliant SVG and PNG NATO symbols using milsymbol.
 *
 * Usage:
 *   npm install milsymbol canvas   (one-time)
 *   node generate_symbols.js
 *
 * Outputs into ./svg/ and ./png/nato/ directories.
 */
const ms = require("milsymbol");
const { createCanvas, loadImage } = require("canvas");
const fs = require("fs");
const path = require("path");

const svgDir = path.join(__dirname, "svg");
const pngDir = path.join(__dirname, "png", "nato");
if (!fs.existsSync(svgDir)) fs.mkdirSync(svgDir, { recursive: true });
if (!fs.existsSync(pngDir)) fs.mkdirSync(pngDir, { recursive: true });

// Affiliation letter in SIDC position 2
const AFFILIATIONS = {
  friendly: "F",
  hostile: "H",
  neutral: "N",
  unknown: "U",
};

// MIL-STD-2525C SIDC templates (replace X with affiliation letter)
// Fighter: function MFF = shows "F" inside frame
// AWACS/AEW: blank air frame + injected "AEW" text
// SAM: ground unit, combat/air defense (UCD)
const ENTITIES = {
  fighter: {
    sidc_template: "SXAPMFF-------", // "F" for fighter
  },
  awacs: {
    sidc_template: "SXAP----------", // blank air frame, we inject "AEW"
    inject_text: "AEW",
    inject_font_size: 25,
  },
  sam: {
    sidc_template: "SXGPUCD-------", // ground air defense unit
  },
};

// Generate PNGs at these sizes
const SIZES = [20, 28, 32, 40, 64, 96, 128];

async function generateAll() {
  let count = 0;

  for (const [entity, cfg] of Object.entries(ENTITIES)) {
    for (const [affiliation, affLetter] of Object.entries(AFFILIATIONS)) {
      const sidc = cfg.sidc_template.replace("X", affLetter);
      const sym = new ms.Symbol(sidc, { size: 80, fill: true, frame: true });
      let svg = sym.asSVG();

      // Inject custom text if needed (e.g. "AEW" for AWACS)
      if (cfg.inject_text) {
        const vb = svg.match(/viewBox="([^"]+)"/)[1].split(" ").map(Number);
        const cx = vb[0] + vb[2] / 2;
        const cy = vb[1] + vb[3] / 2 + 5;
        const fontSize = cfg.inject_font_size || 30;
        const textEl =
          `<text x="${cx}" y="${cy}" text-anchor="middle" ` +
          `font-size="${fontSize}" font-family="Arial" font-weight="bold" ` +
          `dominant-baseline="middle" fill="black">${cfg.inject_text}</text>`;
        svg = svg.replace("</svg>", textEl + "</svg>");
      }

      // Save SVG
      const svgFilename = `${entity}_${affiliation}.svg`;
      fs.writeFileSync(path.join(svgDir, svgFilename), svg);

      // Generate PNGs at each size
      for (const size of SIZES) {
        // Re-generate at the target size for crisp rendering
        const symSmall = new ms.Symbol(sidc, {
          size: size,
          fill: true,
          frame: true,
        });
        let svgSmall = symSmall.asSVG();

        if (cfg.inject_text) {
          const vb = svgSmall
            .match(/viewBox="([^"]+)"/)[1]
            .split(" ")
            .map(Number);
          const cx = vb[0] + vb[2] / 2;
          const cy = vb[1] + vb[3] / 2 + (size * 0.06);
          const fontSize = Math.round(size * 0.45);
          const textEl =
            `<text x="${cx}" y="${cy}" text-anchor="middle" ` +
            `font-size="${fontSize}" font-family="Arial" font-weight="bold" ` +
            `dominant-baseline="middle" fill="black">${cfg.inject_text}</text>`;
          svgSmall = svgSmall.replace("</svg>", textEl + "</svg>");
        }

        const widthMatch = svgSmall.match(/width="([^"]+)"/);
        const heightMatch = svgSmall.match(/height="([^"]+)"/);
        const w = Math.ceil(parseFloat(widthMatch[1]));
        const h = Math.ceil(parseFloat(heightMatch[1]));

        const svgBuffer = Buffer.from(svgSmall);
        const img = await loadImage(svgBuffer);
        const canvas = createCanvas(w, h);
        const ctx = canvas.getContext("2d");
        ctx.drawImage(img, 0, 0, w, h);

        const pngFilename = `${entity}_${affiliation}_${size}.png`;
        fs.writeFileSync(path.join(pngDir, pngFilename), canvas.toBuffer("image/png"));
      }

      console.log(`  ${entity}_${affiliation} (${sidc})`);
      count++;
    }
  }

  console.log(`\nGenerated ${count} NATO symbols (SVG + ${SIZES.length} PNG sizes each)`);
}

// ---------------------------------------------------------------------------
// Flying Objects: rasterise pre-made SVGs from ./flying_objects/
// ---------------------------------------------------------------------------

/**
 * Ensure an SVG string has explicit width and height attributes.
 * The canvas npm module requires these; SVGs that only use viewBox will fail.
 */
function ensureSvgDimensions(svgStr) {
  const hasWidth = /\bwidth\s*=/.test(svgStr);
  const hasHeight = /\bheight\s*=/.test(svgStr);
  if (hasWidth && hasHeight) return svgStr;

  const vbMatch = svgStr.match(/viewBox\s*=\s*["']([^"']+)["']/);
  if (!vbMatch) return svgStr; // Can't infer dimensions; return unchanged

  // viewBox format: "min-x min-y width height"
  const parts = vbMatch[1].trim().split(/[\s,]+/).map(Number);
  const w = parts[2];
  const h = parts[3];

  return svgStr.replace(
    /<svg\b/,
    `<svg width="${w}" height="${h}"`
  );
}

async function generateFlyingObjects() {
  const flyingDir = path.join(__dirname, "flying_objects");
  const pngDir = path.join(__dirname, "png", "flying_objects");
  if (!fs.existsSync(pngDir)) fs.mkdirSync(pngDir, { recursive: true });

  const svgFiles = fs
    .readdirSync(flyingDir)
    .filter((f) => f.endsWith(".svg"));

  let count = 0;
  for (const svgFile of svgFiles) {
    const stem = path.basename(svgFile, ".svg");
    const raw = fs.readFileSync(path.join(flyingDir, svgFile), "utf8");
    const svgStr = ensureSvgDimensions(raw);
    const svgBuffer = Buffer.from(svgStr, "utf8");
    const img = await loadImage(svgBuffer);

    for (const size of SIZES) {
      // Fit inside a size×size canvas while preserving aspect ratio
      const scale = size / Math.max(img.width, img.height);
      const w = Math.ceil(img.width * scale);
      const h = Math.ceil(img.height * scale);
      const offsetX = Math.floor((size - w) / 2);
      const offsetY = Math.floor((size - h) / 2);

      const canvas = createCanvas(size, size);
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, offsetX, offsetY, w, h);

      const pngFilename = `${stem}_${size}.png`;
      fs.writeFileSync(
        path.join(pngDir, pngFilename),
        canvas.toBuffer("image/png")
      );
    }
    console.log(`  ${stem}`);
    count++;
  }
  console.log(
    `\nGenerated flying_objects PNGs for ${count} symbols (${SIZES.length} sizes each)`
  );
}

generateAll()
  .then(() => generateFlyingObjects())
  .catch(console.error);
