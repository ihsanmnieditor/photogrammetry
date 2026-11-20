// Render docs/preview.jpg (1200x630 WhatsApp / social thumbnail) from docs/index.html.
// Usage: serve docs/ on :8767, then  NODE_PATH=$(npm root -g) node scripts/pages_preview.js
const { chromium } = require('playwright');
(async () => {
  const proxy = process.env.HTTPS_PROXY ? { server: process.env.HTTPS_PROXY, bypass: 'localhost,127.0.0.1' } : undefined;
  const b = await chromium.launch({ proxy, args: ['--use-gl=angle', '--use-angle=swiftshader', '--ignore-gpu-blocklist'] });
  const p = await b.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  const errs = []; p.on('console', m => { if (m.type() === 'error') errs.push(m.text().slice(0, 150)); });
  await p.goto('http://localhost:8767/index.html');
  await p.waitForFunction(() => document.querySelector('model-viewer')?.loaded, null, { timeout: 120000 });
  await p.evaluate(() => {
    const mv = document.querySelector('model-viewer');
    mv.autoRotate = false; mv.cameraOrbit = '200deg 72deg 78%'; mv.jumpCameraToGoal();
    document.querySelectorAll('.hint,.controls,.label,.btn.ar').forEach(e => e.style.visibility = 'hidden');
  });
  await p.waitForTimeout(2000);
  await p.screenshot({ path: 'docs/preview.jpg', type: 'jpeg', quality: 88 });
  console.log('errors:', errs.length ? errs : 'none');
  await b.close();
})();
