// index.js for the BiDex-100 project page.
window.HELP_IMPROVE_VIDEOJS = false;

/* ============================================================
 * VLA leaderboard — interactive RoboDojo-style table.
 * No GIF: a sortable, ranked HTML table driven by the data in
 * static/js/lb_chart_data.js (window.LB_CHART_DATA), which is
 * derived from leaderboard_eval_report.json. Toggle the metric
 * (progress score / success rate) and the domain (All = mean of
 * In-Domain & Out-of-Domain, In-Domain, Out-of-Domain); click a
 * column header to re-rank by that axis.
 * ============================================================ */

var LB_D = window.LB_CHART_DATA || {};
// axes from data, plus the derived Overall column
var LB_AXES = (LB_D.axes || []).concat([{ key: 'ov', label: 'Overall' }]);
// models: {html: display name, id: model field key into agg/overall}
var LB_MODELS = (LB_D.models || []).map(function (m) { return { html: m.name, id: m.id }; });

// Live leaderboard state.
var LB = {
  metric: 'score', // 'score' | 'success'
  domain: 'all',   // 'all' (mean of ID & OOD) | 'id' | 'ood'
  sortKey: 'ov',   // axis key to rank by
  sortDir: 'desc'  // 'desc' | 'asc'
};

function lbMetricIdx() { return LB.metric === 'success' ? 1 : 0; }

// row[a-axis] = [score_id, succ_id, score_ood, succ_ood]; overall[f] = {id:[s,u], ood:[s,u], all:[s,u]}
function lbVal(model, axisKey) {
  var mi = lbMetricIdx();
  if (axisKey === 'ov') {
    var ov = LB_D.overall ? LB_D.overall[model.id] : null;
    if (!ov) return 0;
    var d = ov[LB.domain];
    return d ? d[mi] : 0;
  }
  var row = LB_D.agg ? LB_D.agg[model.id] && LB_D.agg[model.id][axisKey] : null;
  if (!row) return 0;
  if (LB.domain === 'all') return (row[mi] + row[mi + 2]) / 2; // mean of ID & OOD (incl. the formula)
  return LB.domain === 'ood' ? row[mi + 2] : row[mi];
}

function lbFmt(v) { return v.toFixed(2); }

// Best value per axis across all models (for the accent highlight).
function lbBestPerAxis() {
  var best = {};
  if (!LB_MODELS.length) return best;
  LB_AXES.forEach(function (a) {
    var m = -Infinity;
    LB_MODELS.forEach(function (mo) {
      var v = lbVal(mo, a.key);
      if (v > m) m = v;
    });
    best[a.key] = m;
  });
  return best;
}

function lbRenderHead() {
  var head = document.getElementById('lb-head-row');
  if (!head) return;
  var html = '<th class="lb-th-rank" aria-label="Rank">#</th>';
  html += '<th class="lb-th-model">Model</th>';
  LB_AXES.forEach(function (a) {
    var sorted = (a.key === LB.sortKey) ? ' is-sorted' : '';
    var overall = (a.key === 'ov') ? ' lb-th-overall' : '';
    var arrow = (a.key === LB.sortKey) ? (LB.sortDir === 'desc' ? ' ▾' : ' ▴') : '';
    html += '<th class="lb-th-metric' + sorted + overall + '" data-lb-sort="' + a.key +
      '" title="' + a.label + '">' +
      '<button class="lb-sort-btn" type="button">' + a.label +
      '<span class="lb-sort-ind">' + arrow + '</span></button></th>';
  });
  head.innerHTML = html;
}

function lbRenderBody() {
  var body = document.getElementById('lb-body');
  if (!body) return;
  if (!LB_D.agg) {
    body.innerHTML = '<tr><td colspan="' + (LB_AXES.length + 2) + '" style="color:var(--muted);padding:1.2rem">Leaderboard data unavailable.</td></tr>';
    return;
  }
  var sorted = LB_MODELS.slice().sort(function (x, y) {
    var d = lbVal(y, LB.sortKey) - lbVal(x, LB.sortKey);
    return LB.sortDir === 'desc' ? d : -d;
  });
  var best = lbBestPerAxis();
  var html = '';
  sorted.forEach(function (mo, i) {
    var rank = i + 1;
    var first = rank === 1 ? ' is-first' : '';
    var row = '<tr class="lb-row' + first + '">';
    row += '<td class="lb-rank' + first + '">' + rank + '</td>';
    row += '<td class="lb-model"><span class="lb-model-name">' + mo.html + '</span></td>';
    LB_AXES.forEach(function (a) {
      var v = lbVal(mo, a.key);
      var leader = (v === best[a.key] && v > 0) ? ' is-leader' : '';
      var overall = (a.key === 'ov') ? ' lb-overall' : '';
      row += '<td class="lb-metric' + leader + overall + '">' + lbFmt(v) + '</td>';
    });
    row += '</tr>';
    html += row;
  });
  body.innerHTML = html;
}

function lbRender() {
  lbRenderHead();
  lbRenderBody();
}

function lbInit() {
  if (!document.getElementById('lb-head-row')) return;

  // Metric chips
  document.querySelectorAll('[data-lb-toggle="metric"]').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('[data-lb-toggle="metric"]').forEach(function (x) {
        x.classList.remove('is-active');
      });
      b.classList.add('is-active');
      LB.metric = b.getAttribute('data-lb-value');
      lbRender();
    });
  });

  // Domain chips
  document.querySelectorAll('[data-lb-toggle="domain"]').forEach(function (b) {
    b.addEventListener('click', function () {
      document.querySelectorAll('[data-lb-toggle="domain"]').forEach(function (x) {
        x.classList.remove('is-active');
      });
      b.classList.add('is-active');
      LB.domain = b.getAttribute('data-lb-value');
      lbRender();
    });
  });

  // Column sort (delegated — the <tr> persists across renders)
  var headRow = document.getElementById('lb-head-row');
  headRow.addEventListener('click', function (e) {
    var th = e.target.closest('th[data-lb-sort]');
    if (!th) return;
    var key = th.getAttribute('data-lb-sort');
    if (LB.sortKey === key) {
      LB.sortDir = (LB.sortDir === 'desc') ? 'asc' : 'desc';
    } else {
      LB.sortKey = key;
      LB.sortDir = 'desc';
    }
    lbRender();
  });

  lbRender();
}

// Boot the leaderboard. We deliberately do NOT depend on jQuery here:
// jQuery is loaded from the Google CDN, which is frequently blocked on
// viewer networks, and a missing `$` would throw in $(document).ready and
// silently prevent the leaderboard + models tables from rendering.
function lbBootstrap() {
  lbInit();

  // jQuery-dependent niceties (navbar burger, carousels) only if jQuery loaded.
  if (!window.$) return;
  $(".navbar-burger").click(function () {
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");
  });
  if (window.bulmaCarousel) {
    bulmaCarousel.attach(".carousel", {
      slidesToScroll: 1, slidesToShow: 2, loop: true, infinite: true,
      autoplay: false, autoplaySpeed: 3000
    });
  }
  if (window.bulmaSlider) { bulmaSlider.attach(); }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', lbBootstrap, { once: true });
} else {
  // DOM already parsed (script ran late / cached) — boot immediately.
  lbBootstrap();
}
