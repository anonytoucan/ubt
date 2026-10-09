// UnicodeMathML (MIT, github.com/MurrayIII/UnicodeMathML) as a Nemeth reader, for scripts/eval/math_baselines.py:
// Nemeth (Unicode braille) -> braille2UnicodeMath -> UnicodeMath -> unicodemathml() -> MathML.
//   node scripts/eval/math_umml.js <UnicodeMathML dir> <in.json: [{id, braille}]> <out.json>
const fs = require('fs'), path = require('path'), vm = require('vm');
const [dir, inPath, outPath] = process.argv.slice(2);
const ctx = {console: {log() {}, warn() {}, error() {}, group() {}, groupEnd() {}}, performance,
             document: {getElementById: () => null, createElement: () => ({})}, navigator: {userAgent: ''}};
ctx.window = ctx; ctx.root = ctx;
vm.createContext(ctx);
for (const f of ['dist/unicodemathml-parser.js', 'src/unicodemathml.js', 'playground/assets/charinfo.js',
                 'playground/assets/TeX.js', 'playground/assets/braille.js'])
  vm.runInContext(fs.readFileSync(path.join(dir, f), 'utf8'), ctx, {filename: f});
const out = JSON.parse(fs.readFileSync(inPath, 'utf8')).map(t => {
  const r = {id: t.id, um: null, mathml: null, err: null};
  try {
    r.um = vm.runInContext('braille2UnicodeMath(' + JSON.stringify(t.braille) + ')', ctx);
    const m = vm.runInContext('unicodemathml(' + JSON.stringify(r.um) + ', false)', ctx);
    if (m && typeof m.mathml === 'string' && m.mathml.startsWith('<math')) r.mathml = m.mathml;
    else r.err = 'unicodemath->mathml failed';
  } catch (e) { r.err = String(e).slice(0, 200); }
  return r;
});
fs.writeFileSync(outPath, JSON.stringify(out));
