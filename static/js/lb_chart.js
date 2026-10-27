/* BiDex-100 per-task bar chart — rendered inline on the main page.
 * ONE large horizontal grouped bar chart: y-axis = the 106 tasks grouped
 * into the five capability axes (colored bands); each task is a row of
 * 7 model bars (one per model, x = 0..100). Toggle metric & domain, hover
 * a bar for the exact value, hover a legend chip to isolate a model.
 * Vanilla SVG, no library/CDN; reads window.LB_CHART_DATA (the same file
 * that drives the leaderboard).
 */
(function () {
  'use strict';
  var D = window.LB_CHART_DATA;
  if (!D || !D.sections) return;

  var ST = { metric: 'score', domain: 'all' };

  // Each per-task model row: [score_id, succ_id, score_ood, succ_ood] (0..100).
  function val(row) {
    var s = ST.metric === 'success' ? 1 : 0;
    var o = ST.metric === 'success' ? 3 : 2;
    var a = row[s], b = row[o];
    if (ST.domain === 'all') {
      if (a == null) return b;
      if (b == null) return a;
      return (a + b) / 2;
    }
    return ST.domain === 'ood' ? b : a;
  }
  function fmtPct(v) { return v == null ? '—' : Number(v).toFixed(2) + '%'; }

  // geometry
  var LABEL_W = 184, PLOT_W = 720, RK = 14, W = LABEL_W + PLOT_W + RK;
  var ROW_PITCH = 26, BAR_H = 3.1, BAR_GAP = 0.45, GPAD = 0.8;
  var HEADER_H = 24, SEC_GAP = 10, TOP_PAD = 6;

  function esc(s) {
    return String(s).replace(/[<>&"']/g, function (c) {
      return ({ '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;' })[c];
    });
  }
  function trunc(s, n) { s = String(s); return s.length > n ? s.slice(0, n - 1) + '…' : s; }

  function axisStrip(height) {
    var g = '';
    [0, 25, 50, 75, 100].forEach(function (p) {
      var x = LABEL_W + PLOT_W * (p / 100);
      g += '<line class="lbch-vgrid" x1="' + x + '" x2="' + x + '" y1="0" y2="' + height + '"/>';
    });
    return g;
  }

  function buildChart() {
    // aggregate height
    var H = TOP_PAD;
    var secTop = []; // y of each section band
    D.sections.forEach(function (s) { H += HEADER_H + s.tasks.length * ROW_PITCH + SEC_GAP; });
    H += TOP_PAD;

    // vertical gridlines spanning whole chart (behind bars)
    var grid = axisStrip(H);
    // baseline at bar start (x = LABEL_W)
    grid += '<line class="lbch-vbase" x1="' + LABEL_W + '" x2="' + LABEL_W + '" y1="0" y2="' + H + '"/>';

    var cursor = TOP_PAD;
    var body = '';
    D.sections.forEach(function (s) {
      var bandH = HEADER_H + s.tasks.length * ROW_PITCH;
      var yTop = cursor;
      secTop.push(yTop);
      // category band (faint) + left accent
      body += '<rect class="lbch-band" x="0" y="' + yTop + '" width="' + W + '" height="' + bandH +
        '" fill="' + s.color + '" fill-opacity="0.06"/>';
      body += '<rect class="lbch-band-edge" x="0" y="' + yTop + '" width="5" height="' + bandH + '" fill="' + s.color + '"/>';
      // category title
      body += '<text class="lbch-cat" x="14" y="' + (yTop + 16) + '" fill="' + s.color + '">' +
        esc(s.name) + ' <tspan class="lbch-cat-n">(' + s.n + ' tasks)</tspan></text>';

      var rowTop = yTop + HEADER_H;
      s.tasks.forEach(function (task, ti) {
        var y = rowTop + ti * ROW_PITCH;
        // task label (left, truncated)
        body += '<text class="lbch-tlabel" x="14" y="' + (y + ROW_PITCH / 2 + 3) + '"' +
          ' data-taskf="' + esc(task.name) + '">' + esc(trunc(task.full || task.name, 30)) + '</text>';
        // 7 model bars
        task.v.forEach(function (row, mi) {
          var v = val(row);
          if (v == null) return;
          var w = PLOT_W * (v / 100);
          var by = y + GPAD + mi * (BAR_H + BAR_GAP);
          var fill = D.models[mi].color;
          body += '<rect class="lbch-bar lbch-mi-' + mi + '" data-mi="' + mi +
            '" data-model="' + esc(D.models[mi].name) + '" data-task="' + esc(task.name) +
            '" data-val="' + Number(v).toFixed(2) + '" x="' + LABEL_W + '" y="' + by.toFixed(2) +
            '" width="' + w.toFixed(2) + '" height="' + BAR_H + '" rx="1.1" fill="' + fill + '"/>';
        });
      });
      cursor = yTop + bandH + SEC_GAP;
    });

    return '<svg class="lbch-svg" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H +
      '" preserveAspectRatio="xMinYMin meet" role="img" aria-label="per-task performance by model">' +
      '<rect class="lbch-bg" x="0" y="0" width="' + W + '" height="' + H + '"/>' + grid + body + '</svg>';
  }

  function axisHeader() {
    var ticks = '';
    [0, 25, 50, 75, 100].forEach(function (p) {
      var x = LABEL_W + PLOT_W * (p / 100);
      ticks += '<text class="lbch-ax-x" x="' + x + '" y="15" text-anchor="middle">' + p + '</text>';
    });
    return '<svg class="lbch-axis" width="' + W + '" height="20" viewBox="0 0 ' + W + ' 20" preserveAspectRatio="xMinYMin meet">' + ticks + '</svg>';
  }

  function buildTop() {
    var chips = function (kind, vals, active) {
      return vals.map(function (v) {
        return '<button class="lb-chip' + (v.v === active ? ' is-active' : '') +
          '" data-ch="' + kind + '" data-ch-v="' + v.v + '" type="button">' + v.label + '</button>';
      }).join('');
    };
    var legend = D.models.map(function (m, i) {
      return '<li class="lbch-legend-item" data-mi="' + i + '"' +
        ' style="border-left:3px solid ' + m.color + '">' +
        '<span class="lbch-swatch" style="background:' + m.color + '"></span>' +
        '<span class="lbch-legend-name">' + esc(m.name) + '</span></li>';
    }).join('');
    var metricLbl = ST.metric === 'success' ? 'Strict success rate (%)' : 'Progress score (%)';
    var domainLbl = ST.domain === 'all' ? 'All (mean of In-Domain & Out-of-Domain)' : (ST.domain === 'id' ? 'In-Domain' : 'Out-of-Domain');
    return (
      '<div class="lbch-head">' +
        '<p class="lb-kicker">BiDex-100 · Per-Task Performance</p>' +
        '<h2 class="lbch-title">All 106 tasks × 7 VLA models</h2>' +
        '<p class="lbch-sub">One bar per (task, model): ' + metricLbl + ' · ' + domainLbl +
        '. Tasks are grouped by capability axis (colored). Hover a bar for the exact value; hover a model name to isolate it.</p>' +
      '</div>' +
      '<div class="lbch-controls lb-toolbar">' +
        '<div class="lb-toolbar-group">' +
          '<span class="lb-sort-label">Metric</span>' +
          '<div class="lb-chips" role="group" aria-label="Metric">' +
            chips('metric', [{ v: 'score', label: 'Progress Score' }, { v: 'success', label: 'Success Rate' }], ST.metric) +
          '</div>' +
        '</div>' +
        '<div class="lb-toolbar-group">' +
          '<span class="lb-sort-label">Domain</span>' +
          '<div class="lb-chips" role="group" aria-label="Domain">' +
            chips('domain', [{ v: 'all', label: 'All' }, { v: 'id', label: 'In-Domain' }, { v: 'ood', label: 'Out-of-Domain' }], ST.domain) +
          '</div>' +
        '</div>' +
      '</div>' +
      '<ul class="lbch-legend">' + legend + '</ul>'
    );
  }

  var root, tip;
  function render() {
    root.innerHTML = buildTop() +
      '<div class="lbch-stage">' +
        '<div class="lbch-ax-sticky">' + axisHeader() + '</div>' +
        '<div class="lbch-svg-host">' + buildChart() + '</div>' +
      '</div>';
    wire();
  }

  function wire() {
    document.querySelectorAll('[data-ch]').forEach(function (b) {
      b.addEventListener('click', function () {
        var kind = b.getAttribute('data-ch');
        document.querySelectorAll('[data-ch="' + kind + '"]').forEach(function (x) { x.classList.remove('is-active'); });
        b.classList.add('is-active');
        ST[kind] = b.getAttribute('data-ch-v');
        render();
      });
    });
    root.querySelectorAll('.lbch-legend-item').forEach(function (li) {
      var mi = li.getAttribute('data-mi');
      li.addEventListener('mouseenter', function () { root.classList.add('lbch-focus-mi-' + mi); });
      li.addEventListener('mouseleave', function () { root.classList.remove('lbch-focus-mi-' + mi); });
    });
    root.querySelectorAll('.lbch-bar').forEach(function (r) {
      r.addEventListener('mouseenter', function (e) {
        showTip(e.clientX, e.clientY, r.getAttribute('data-model'), r.getAttribute('data-task'), parseFloat(r.getAttribute('data-val')));
      });
      r.addEventListener('mouseleave', hideTip);
      r.addEventListener('mousemove', function (e) {
        tip.style.left = (e.clientX + 14) + 'px';
        tip.style.top = (e.clientY - 14 - (tip.offsetHeight || 48)) + 'px';
      });
    });
  }

  function showTip(cx, cy, model, task, v) {
    if (!tip) return;
    var short = task.split('/').pop().replace(/_/g, ' ');
    tip.innerHTML = '<div class="lbch-tip-model">' + esc(model) + '</div>' +
      '<div class="lbch-tip-task" title="' + esc(task) + '">' + esc(short) + '</div>' +
      '<div class="lbch-tip-val">' + (ST.metric === 'success' ? 'Success' : 'Score') +
      ' · ' + (ST.domain === 'all' ? 'All' : ST.domain === 'id' ? 'ID' : 'OOD') + ': <b>' + fmtPct(v) + '</b></div>';
    tip.style.display = 'block';
    var w = tip.offsetWidth || 200, h = tip.offsetHeight || 52;
    var x = cx + 14, y = cy - h - 14;
    if (x + w > window.innerWidth - 8) x = window.innerWidth - w - 8;
    if (y < 8) y = cy + 16;
    tip.style.left = x + 'px';
    tip.style.top = y + 'px';
  }
  function hideTip() { if (tip) tip.style.display = 'none'; }

  function init() {
    root = document.getElementById('lbch-root');
    if (!root) return;
    tip = document.getElementById('lbch-tip');
    render();
    // keep tooltip geometry correct on resize (reposition handled per-move)
    window.addEventListener('scroll', hideTip, { passive: true });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else { init(); }
})();
