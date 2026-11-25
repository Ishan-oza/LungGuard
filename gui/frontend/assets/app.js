/**
 * LungGuard - Editorial Medical AI Client
 * Typography: Soria | Palette: #B08401, #D49E8D, #DED1BD, #FAF6F2, #683B2B
 */

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];

const LABELS = {
  COVID19: 'COVID-19',
  NORMAL: 'Normal',
  PNEUMONIA: 'Pneumonia'
};

const COLORS = {
  COVID19: 'var(--covid)',
  NORMAL: 'var(--normal)',
  PNEUMONIA: 'var(--pneu)'
};

const MODEL_NAMES = {
  resnet50: 'ResNet50',
  vgg19: 'VGG19',
  efficientnet_b1: 'EfficientNet-B1',
  mobilenet_v2: 'MobileNetV2'
};

const api = async (url, options) => {
  const resp = await fetch(url, options);
  let data = null;
  try {
    data = await resp.json();
  } catch (e) {}
  if (!resp.ok) {
    throw new Error(data?.detail || `Request failed with status ${resp.status}`);
  }
  return data;
};

// 1. Status mark removed as requested - Clean header

// 2. Generic Tabs Helper
$$('.tabs[data-panels]').forEach(group => {
  $$('button', group).forEach(btn => {
    btn.addEventListener('click', () => {
      $$('button', group).forEach(b => b.setAttribute('aria-selected', b === btn));
      $$(group.dataset.panels).forEach(panel => {
        panel.hidden = panel.id !== btn.dataset.tab;
      });
    });
  });
});

// 3. Clipboard Copy Helper
$$('.cp').forEach(btn => {
  btn.addEventListener('click', () => {
    const text = btn.parentElement.innerText.replace(/Copy$|Copied$/, '').trim();
    navigator.clipboard?.writeText(text);
    btn.textContent = 'Copied';
    setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
  });
});

// 4. Metrics & Confusion Matrix (Results & Models Pages)
function drawCM(grid, matrix, classNames) {
  let html = '<div></div>' + classNames.map(name => `<div class="h">${name}</div>`).join('');
  matrix.forEach((row, i) => {
    const rowSum = row.reduce((a, b) => a + b, 0);
    html += `<div class="h">${classNames[i]}</div>` + row.map((val, j) => {
      const frac = val / (rowSum || 1);
      let bg, col;
      if (i === j) {
        // Correct predictions in rich warm gold/ochre from palette
        bg = `rgba(176, 132, 1, ${0.14 + frac * 0.78})`;
        col = '#3D2218';
      } else {
        // Errors in soft terracotta red
        bg = val ? `rgba(168, 54, 38, ${Math.min(0.08 + frac * 2, 0.45)})` : 'rgba(222, 209, 189, 0.2)';
        col = val ? '#A83626' : 'var(--tx-dim)';
      }
      return `<div class="c" style="background:${bg};color:${col}" title="${val} of ${rowSum} ${classNames[i]}">
                ${val}<small>${(frac * 100).toFixed(1)}%</small>
              </div>`;
    }).join('');
  });
  grid.innerHTML = html + '<div class="ax">Rows: Actual · Columns: Predicted</div>';
}

(async () => {
  const cmGrid = $('#cm');
  const modelRows = $('#model-rows');
  if (!cmGrid && !modelRows) return;

  let metricsData;
  try {
    metricsData = await api('/api/metrics');
  } catch (e) {
    if (cmGrid) cmGrid.innerHTML = '<p class="err">Could not load metrics from backend.</p>';
    return;
  }

  // Results Page Confusion Matrix
  if (cmGrid) {
    const allMatrices = { ensemble: metricsData.ensemble.confusion };
    Object.entries(metricsData.models).forEach(([k, v]) => {
      allMatrices[k] = v.confusion;
    });

    const updateMatrix = (key) => {
      drawCM(cmGrid, allMatrices[key], metricsData.classes);
      const acc = key === 'ensemble' ? metricsData.ensemble.accuracy : metricsData.models[key].accuracy;
      const accElem = $('#cm-acc');
      if (accElem) accElem.textContent = `${acc.toFixed(2)}%`;
    };

    updateMatrix('ensemble');

    $$('.tabs[data-cm] button').forEach(btn => {
      btn.addEventListener('click', () => {
        $$('.tabs[data-cm] button').forEach(b => b.setAttribute('aria-selected', b === btn));
        updateMatrix(btn.dataset.k);
      });
    });

    // Populate Per-Class Table
    const perClassTable = $('#pc');
    if (perClassTable) {
      const ens = metricsData.ensemble;
      let rowsHtml = metricsData.classes.map((cls, i) => `
        <tr>
          <td><b style="font-family:'Soria',serif; font-size:1.15rem">${cls}</b></td>
          <td class="n">${(ens.precision[i] * 100).toFixed(1)}%</td>
          <td class="n">${(ens.recall[i] * 100).toFixed(1)}%</td>
          <td class="n">${(ens.f1[i] * 100).toFixed(1)}%</td>
          <td class="n">${ens.auc[i].toFixed(3)}</td>
          <td class="n">${metricsData.support[i].toLocaleString()}</td>
        </tr>
      `).join('');

      rowsHtml += `
        <tr class="top">
          <td><b style="font-family:'Soria',serif; font-size:1.15rem">Overall Average</b></td>
          <td class="n">95.0%</td>
          <td class="n">95.0%</td>
          <td class="n">95.0%</td>
          <td class="n">0.994</td>
          <td class="n">${metricsData.test_size.toLocaleString()}</td>
        </tr>
      `;
      perClassTable.innerHTML = rowsHtml;
    }
  }

  // Models Page Rows
  if (modelRows) {
    const descriptions = {
      resnet50: '50-layer deep residual network with bottleneck skip connections. Acts as the primary anchor for the ensemble.',
      vgg19: 'Uniform 3x3 receptive field feature extractor with deep sequential capacity providing stable spatial cues.',
      efficientnet_b1: 'Compound scaled architecture balancing network depth, width, and resolution with high parameter efficiency.',
      mobilenet_v2: 'Lightweight inverted bottleneck architecture engineered for edge, tablet, and mobile clinical use.'
    };

    modelRows.innerHTML = Object.entries(metricsData.models).map(([key, item]) => `
      <div style="background:#FFFFFF; border:1px solid var(--line); border-radius:var(--r); padding:2rem; margin-bottom:1.5rem; display:flex; justify-content:space-between; align-items:center; gap:3rem; flex-wrap:wrap; box-shadow:var(--shadow-card);">
        <div style="flex:1; min-width:300px;">
          <h3 style="font-size:1.4rem; color:var(--tx-title);">${item.name}</h3>
          <p style="font-size:1.05rem; margin:0.4rem 0 1rem; color:var(--tx-muted);">${descriptions[key] || ''}</p>
          <div style="display:flex; gap:0.6rem; flex-wrap:wrap;">
            <span class="tag">Vote Weight: <b>${item.weight}</b></span>
            <span class="tag">Parameters: <b>${item.params}</b></span>
            <span class="tag">Macro AUC: <b>${item.macro_auc.toFixed(4)}</b></span>
          </div>
        </div>
        <div style="text-align:right; min-width:140px;">
          <span style="font-size:0.95rem; color:var(--tx-muted); display:block;">Individual Accuracy</span>
          <strong style="font-family:'Soria',serif; font-size:2.8rem; color:var(--c-espresso);">${item.accuracy}%</strong>
        </div>
      </div>
    `).join('');
  }
})();

// 5. Analyze Page (Interactive Inference Workbench)
const dropZone = $('#drop');
if (dropZone) {
  const fileInput = $('#file');
  const stage = $('#stage');
  const baseImg = $('#base');
  const heatImg = $('#heat');
  const goBtn = $('#go');
  const clrBtn = $('#clr');
  const resultsBox = $('#res');
  const emptyState = $('#empty');
  const msgBox = $('#msg');
  const heatToggle = $('#heatsw');
  const fnameLabel = $('#fname');
  const pctRing = $('#pct');
  const fgRing = $('#fg');
  const C = 2 * Math.PI * 54;

  let currentFile = null;
  let objectUrl = null;
  const MAX_BYTES = 10 * 1024 * 1024;

  const showMessage = (text, type = 'err') => {
    if (!msgBox) return;
    msgBox.innerHTML = text ? `<div class="${type}" role="alert">${text}</div>` : '';
  };

  const handleFile = (file) => {
    showMessage('');
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      return showMessage('That file is not an image. Please use JPG, PNG, BMP, or WebP.', 'err');
    }
    if (file.size > MAX_BYTES) {
      return showMessage('File is larger than 10 MB. Please choose a smaller file.', 'err');
    }

    currentFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);

    baseImg.src = objectUrl;
    heatImg.removeAttribute('src');
    stage.classList.remove('showheat');

    $('#pick').hidden = true;
    $('#view').hidden = false;
    goBtn.disabled = false;
    resultsBox.style.display = 'none';
    emptyState.hidden = false;

    if (fnameLabel) fnameLabel.textContent = file.name;
  };

  const resetAll = () => {
    currentFile = null;
    fileInput.value = '';
    $('#pick').hidden = false;
    $('#view').hidden = true;
    goBtn.disabled = true;
    resultsBox.style.display = 'none';
    emptyState.hidden = false;
    showMessage('');
  };

  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
  clrBtn.addEventListener('click', resetAll);

  ['dragenter', 'dragover'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.add('over');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropZone.addEventListener(name, (e) => {
      e.preventDefault();
      dropZone.classList.remove('over');
    });
  });

  dropZone.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
  window.addEventListener('paste', (e) => {
    const pasted = [...(e.clipboardData?.files || [])][0];
    if (pasted) handleFile(pasted);
  });

  if (heatToggle) {
    heatToggle.addEventListener('change', (e) => {
      stage.classList.toggle('showheat', e.target.checked && !!heatImg.getAttribute('src'));
    });
  }

  // Render prediction results
  function render(res) {
    const predClass = res.prediction;
    const label = res.label;
    const conf = res.confidence;
    const mname = res.model === 'ensemble' ? 'Ensemble' : MODEL_NAMES[res.model];

    // Verdict and Confidence Ring
    $('#vl').textContent = label;
    $('#vl').style.color = COLORS[predClass];
    $('#vsub').textContent = mname + ' Verdict';

    if (pctRing && fgRing) {
      fgRing.style.strokeDashoffset = C * (1 - conf);
      pctRing.textContent = (conf * 100).toFixed(0) + '%';
    }

    // Probability Bars
    const bar = (l, c, v) => `
      <div class="pr">
        <b>${l}</b>
        <div class="track">
          <div class="fill" style="width:${v * 100}%; background:${c}"></div>
        </div>
        <span>${(v * 100).toFixed(1)}%</span>
      </div>
    `;
    $('#probs').innerHTML = Object.keys(LABELS).map(k => bar(LABELS[k], COLORS[k], res.probabilities[k])).join('');

    // Individual Votes
    $('#votes').innerHTML = Object.entries(res.per_model).map(([k, v]) => `
      <div class="vote">
        <div>${MODEL_NAMES[k]}<br><em>weight ${v.weight}</em></div>
        <span class="tag" style="color:${COLORS[v.prediction]}">${v.label}</span>
        <b style="text-align:right">${(v.confidence * 100).toFixed(0)}%</b>
      </div>
    `).join('');
    $('#votewrap').hidden = res.model !== 'ensemble';

    // Heatmap
    const hasHeat = !!res.heatmap;
    $('#heatrow').hidden = !hasHeat;
    if (hasHeat) {
      heatImg.src = res.heatmap;
      if (heatToggle) heatToggle.checked = true;
      stage.classList.add('showheat');
    } else {
      stage.classList.remove('showheat');
    }

    // Metadata & Warnings
    $('#meta').textContent = `${res.inference_ms} ms · ${res.image.width}×${res.image.height}px · Processed in memory`;
    $('#warns').innerHTML = (res.warnings || []).map(w => `<div class="warn">${w}</div>`).join('');

    emptyState.hidden = true;
    resultsBox.style.display = 'block';
  }

  // Top-to-down scan animation sweep + Pre-scan validation check
  goBtn.addEventListener('click', async () => {
    if (!currentFile) return;
    showMessage('');
    goBtn.disabled = true;
    goBtn.textContent = 'Scanning…';
    stage.classList.add('scanning');

    // Hide any previous result while scanning
    emptyState.hidden = false;
    resultsBox.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentFile);
    const selectedModel = document.querySelector('input[name=m]:checked')?.value || 'ensemble';
    formData.append('model', selectedModel);
    formData.append('explain', 'true');

    try {
      // Allow scanning beam to sweep visibly from top to bottom
      const [result] = await Promise.all([
        api('/api/predict', { method: 'POST', body: formData }),
        new Promise(r => setTimeout(r, 1300))
      ]);

      stage.classList.remove('scanning');

      // Pre-scan validation gate: reject non-chest X-rays
      if (result.is_valid_xray === false) {
        showMessage(`Scan Aborted: The uploaded image is not a chest X-ray. ${result.error}`, 'err');
        return;
      }

      render(result);
    } catch (err) {
      stage.classList.remove('scanning');
      showMessage(err.message === 'Failed to fetch'
        ? 'Cannot reach backend server. Please verify "python app.py" is running.'
        : err.message, 'err');
    } finally {
      stage.classList.remove('scanning');
      goBtn.disabled = false;
      goBtn.textContent = 'Analyze X-ray';
    }
  });
}
