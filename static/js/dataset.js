/* Dataset metadata visualizations; no external chart runtime. */
(function () {
  "use strict";
  const root = document.querySelector('#dataset-charts');
  if (!root) return;
  const fmt = n => Number(n).toLocaleString('en-US');
  const ns = 'http://www.w3.org/2000/svg';
  const svgEl = (tag, attrs = {}, text) => {
    const node = document.createElementNS(ns, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  function text(svg, x, y, label, cls, anchor) {
    const node = svgEl('text', {x, y, ...(cls ? {class: cls} : {}), ...(anchor ? {'text-anchor': anchor} : {})}, label);
    svg.append(node); return node;
  }
  fetch('static/data/dataset-statistics.json').then(response => {
    if (!response.ok) throw new Error('Dataset statistics unavailable');
    return response.json();
  }).then(data => {
    const axes = data.axes;
    const axisMap = Object.fromEntries(axes.map(axis => [axis.key, axis]));
    const robotMap = Object.fromEntries(data.robots.map(robot => [robot.key, robot]));
    const flowHost = document.querySelector('#dataset-flow');
    const flowReadout = document.querySelector('#flow-readout');
    const compositionReadout = document.querySelector('#composition-readout');
    let metric = 'tasks';
    let selectedAxis = null;

    let flowAxis = 'all';
    let flowWeight = 'demonstrations';
    let pinnedTask = null;
    const durationBands = [
      {key: 'short', label: 'Under 20 s', limit: 20, color: '#91a59e'},
      {key: 'medium', label: '20–40 s', limit: 40, color: '#758f85'},
      {key: 'long', label: '40–60 s', limit: 60, color: '#53786b'},
      {key: 'extended', label: '60 s and above', limit: Infinity, color: '#345e51'}
    ];
    const bandOf = task => durationBands.find(band => task.mean_duration_seconds < band.limit).key;
    const names = Object.fromEntries((window.BENCH?.TASKS || []).map(task => [task.axis + '/' + task.id, task.name]));
    const taskName = task => names[task.key] || task.name.replaceAll('_', ' ').replace(/\b\w/g, letter => letter.toUpperCase());
    const weightLabel = value => flowWeight === 'hours' ? `${value.toFixed(2)} h` : fmt(value);

    function renderFlow() {
      const tasks = data.tasks.filter(task => flowAxis === 'all' || task.axis === flowAxis);
      const axisKeys = axes.filter(axis => tasks.some(task => task.axis === axis.key));
      const robots = data.robots.filter(robot => tasks.some(task => task.robot === robot.key));
      const bands = durationBands.filter(band => tasks.some(task => bandOf(task) === band.key));
      const total = tasks.reduce((sum, task) => sum + task[flowWeight], 0);
      const W = 1120;
      const H = flowAxis === 'all' ? 720 : Math.max(550, tasks.length * 18 + 92);
      const top = 54, available = H - 94;
      const taskGap = flowAxis === 'all' ? 1.5 : 7;
      const scale = (available - taskGap * (tasks.length - 1)) / total;
      const svg = svgEl('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-labelledby': 'dataset-flow-title dataset-flow-desc'});
      svg.append(svgEl('title', {id: 'dataset-flow-title'}, 'Embodiment, capability, task and mean-duration flows'));
      svg.append(svgEl('desc', {id: 'dataset-flow-desc'}, `${tasks.length} tasks. Flow width represents ${flowWeight === 'hours' ? 'recorded hours' : 'demonstrations'}. Duration bands use each task’s mean demonstration length, not individual episode lengths. Select a task to trace its source and duration band.`));
      const sum = rows => rows.reduce((value, task) => value + task[flowWeight], 0);
      const columns = [
        robots.map(robot => ({id: 'robot:' + robot.key, key: robot.key, label: robot.label, value: sum(tasks.filter(task => task.robot === robot.key)), color: robot.key === data.robots[0].key ? '#1a7a6d' : '#a47c35', rows: tasks.filter(task => task.robot === robot.key)})),
        axisKeys.map(axis => ({id: 'axis:' + axis.key, key: axis.key, label: axis.label, value: sum(tasks.filter(task => task.axis === axis.key)), color: axis.color, rows: tasks.filter(task => task.axis === axis.key)})),
        axisKeys.flatMap(axis => tasks.filter(task => task.axis === axis.key).map(task => ({id: 'task:' + task.key, key: task.key, label: taskName(task), value: task[flowWeight], color: axis.color, rows: [task]}))),
        bands.map(band => ({id: 'band:' + band.key, key: band.key, label: band.label, value: sum(tasks.filter(task => bandOf(task) === band.key)), color: band.color, rows: tasks.filter(task => bandOf(task) === band.key)}))
      ];
      const xPositions = [34, 326, 642, 942];
      const gaps = [34, 25, taskGap, 30];
      const lookup = {};
      columns.forEach((column, index) => {
        const gap = gaps[index];
        let y = top + (available - total * scale - gap * (column.length - 1)) / 2;
        column.forEach(node => {
          Object.assign(node, {x: xPositions[index], y, height: node.value * scale, outgoing: 0, incoming: 0});
          y += node.height + gap;
          lookup[node.id] = node;
        });
      });
      const linkSpecs = [];
      robots.forEach(robot => axisKeys.forEach(axis => {
        const rows = tasks.filter(task => task.robot === robot.key && task.axis === axis.key);
        if (rows.length) linkSpecs.push({from: 'robot:' + robot.key, to: 'axis:' + axis.key, rows, color: axis.color});
      }));
      columns[2].forEach(node => {
        const task = node.rows[0];
        linkSpecs.push({from: 'axis:' + task.axis, to: node.id, rows: [task], color: node.color});
        linkSpecs.push({from: node.id, to: 'band:' + bandOf(task), rows: [task], color: node.color});
      });
      const links = svgEl('g', {class: 'flow-links'}); svg.append(links);
      const showTask = task => {
        const related = node => (node.getAttribute('data-tasks') || '').split('|').includes(task.key);
        svg.classList.add('has-selection');
        svg.querySelectorAll('.flow-link,.flow-node-group').forEach(node => node.classList.toggle('is-related', related(node)));
        flowReadout.textContent = `${robotMap[task.robot].label} → ${axisMap[task.axis].label} → ${taskName(task)} → ${durationBands.find(band => band.key === bandOf(task)).label}`;
        document.querySelector('#flow-task-name').textContent = taskName(task);
        document.querySelector('#flow-task-stats').textContent = `${fmt(task.demonstrations)} demonstrations · ${fmt(task.frames)} frames · ${task.hours.toFixed(2)} h recorded · ${task.mean_duration_seconds.toFixed(1)} s mean demonstration`;
      };
      const clearSelection = () => {
        if (pinnedTask) {showTask(pinnedTask); return;}
        svg.classList.remove('has-selection');
        flowReadout.textContent = `${robots.length} embodiment${robots.length === 1 ? '' : 's'} → ${axisKeys.length} capability ${axisKeys.length === 1 ? 'axis' : 'axes'} → ${tasks.length} tasks → ${bands.length} mean-duration bands. Width = ${flowWeight === 'hours' ? 'recorded hours' : 'demonstrations'}.`;
        document.querySelector('#flow-task-name').textContent = 'Trace an individual task';
        document.querySelector('#flow-task-stats').textContent = 'Hover or focus to inspect. Click a task to keep its path highlighted; click again to release.';
      };
      linkSpecs.forEach(link => {
        const source = lookup[link.from], target = lookup[link.to];
        const value = sum(link.rows), h = value * scale;
        const a = source.y + source.outgoing, b = target.y + target.incoming;
        const x1 = source.x + 12, x2 = target.x, mid = (x1 + x2) / 2;
        const d = `M ${x1} ${a} C ${mid} ${a} ${mid} ${b} ${x2} ${b} L ${x2} ${b + h} C ${mid} ${b + h} ${mid} ${a + h} ${x1} ${a + h} Z`;
        const label = `${source.label} → ${target.label}: ${weightLabel(value)} ${flowWeight === 'hours' ? 'recorded' : 'demonstrations'}`;
        const path = svgEl('path', {d, fill: link.color, class: 'flow-link', 'data-tasks': link.rows.map(task => task.key).join('|'), 'aria-label': label});
        path.append(svgEl('title', {}, label));
        if (link.rows.length === 1) {
          path.addEventListener('pointerenter', () => {if (!pinnedTask) showTask(link.rows[0]);});
          path.addEventListener('click', () => {pinnedTask = pinnedTask?.key === link.rows[0].key ? null : link.rows[0]; clearSelection();});
        } else path.addEventListener('pointerenter', () => {if (!pinnedTask) flowReadout.textContent = label;});
        path.addEventListener('pointerleave', clearSelection);
        links.append(path); source.outgoing += h; target.incoming += h;
      });
      const headings = ['EMBODIMENT', 'CAPABILITY AXIS', `${tasks.length} INDIVIDUAL TASKS`, 'TASK MEAN DURATION'];
      headings.forEach((heading, index) => text(svg, xPositions[index], 23, heading, 'flow-heading'));
      columns.forEach((column, columnIndex) => {
        let lastLabelY = -100;
        column.forEach(node => {
          const group = svgEl('g', {class: 'flow-node-group', 'data-tasks': node.rows.map(task => task.key).join('|')});
          const rect = svgEl('rect', {x: node.x, y: node.y, width: 12, height: node.height, fill: node.color, class: 'flow-node'});
          group.append(rect); svg.append(group);
          const cy = node.y + node.height / 2;
          if (columnIndex === 2) {
            const task = node.rows[0];
            group.setAttribute('tabindex', '0'); group.setAttribute('role', 'button');
            group.setAttribute('aria-label', `${node.label}, ${fmt(task.demonstrations)} demonstrations, ${task.mean_duration_seconds.toFixed(1)} seconds mean duration. Select to trace task.`);
            group.append(svgEl('rect', {x: node.x - 5, y: Math.max(top, cy - 5), width: flowAxis === 'all' ? 24 : 270, height: 10, fill: 'transparent', class: 'flow-task-hit'}));
            group.append(svgEl('title', {}, group.getAttribute('aria-label')));
            group.addEventListener('pointerenter', () => {if (!pinnedTask) showTask(task);});
            group.addEventListener('pointerleave', clearSelection);
            group.addEventListener('focus', () => showTask(task));
            group.addEventListener('blur', clearSelection);
            const select = () => {pinnedTask = pinnedTask?.key === task.key ? null : task; clearSelection();};
            group.addEventListener('click', select);
            group.addEventListener('keydown', event => {
              if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); select();}
              if (event.key === 'Escape') {pinnedTask = null; clearSelection();}
            });
            if (flowAxis !== 'all' && cy - lastLabelY > 13) {
              const label = text(group, node.x + 21, cy + 3.5, node.label.length > 35 ? node.label.slice(0, 33) + '…' : node.label, 'flow-task-label');
              label.append(svgEl('title', {}, node.label)); lastLabelY = cy;
            }
          } else {
            const label = text(group, node.x + 20, cy - 4, node.label, 'flow-node-label');
            if (columnIndex === 1 && node.label.length > 20) label.style.fontSize = '11px';
            text(group, node.x + 20, cy + 15, weightLabel(node.value), 'flow-count');
            const caption = text(group, node.x + 20, cy + 31, `${node.rows.length} tasks`, 'flow-task-total');
            caption.style.fontSize = '10px';
          }
        });
      });
      flowHost.replaceChildren(svg);
      document.querySelector('#flow-visible-total').textContent = `${tasks.length} tasks · ${flowWeight === 'hours' ? total.toFixed(2) + ' recorded hours' : fmt(total) + ' demonstrations'}`;
      clearSelection();
    }
    document.querySelectorAll('[data-flow-axis]').forEach(button => button.addEventListener('click', () => {
      flowAxis = button.dataset.flowAxis; pinnedTask = null;
      document.querySelectorAll('[data-flow-axis]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
      renderFlow();
    }));
    document.querySelectorAll('[data-flow-weight]').forEach(button => button.addEventListener('click', () => {
      flowWeight = button.dataset.flowWeight; pinnedTask = null;
      document.querySelectorAll('[data-flow-weight]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
      renderFlow();
    }));

    function polar(angle, radius) {return [110 + radius * Math.cos(angle), 110 + radius * Math.sin(angle)];}
    function segment(start, end) {
      const a = polar(start, 92), b = polar(end, 92), c = polar(end, 64), d = polar(start, 64);
      const large = end - start > Math.PI ? 1 : 0;
      return `M ${a} A 92 92 0 ${large} 1 ${b} L ${c} A 64 64 0 ${large} 0 ${d} Z`;
    }
    function renderComposition() {
      const total = data.totals[metric];
      const svg = svgEl('svg', {viewBox: '0 0 220 220', role: 'img', 'aria-labelledby': 'dataset-donut-title'});
      svg.append(svgEl('title', {id: 'dataset-donut-title'}, `Capability shares by ${metric}`));
      let angle = -Math.PI / 2;
      axes.forEach(axis => {
        const next = angle + axis[metric] / total * Math.PI * 2;
        const node = svgEl('path', {d: segment(angle + .012, next - .012), fill: axis.color, class: 'donut-segment', tabindex: '0', 'data-axis': axis.key, 'aria-label': `${axis.label}: ${fmt(axis[metric])} ${metric}, ${(100 * axis[metric] / total).toFixed(1)} percent`});
        node.append(svgEl('title', {}, node.getAttribute('aria-label')));
        node.addEventListener('pointerenter', () => highlight(axis.key));
        node.addEventListener('focus', () => highlight(axis.key));
        node.addEventListener('click', () => highlight(selectedAxis === axis.key ? null : axis.key));
        node.addEventListener('keydown', event => {if (event.key === 'Enter' || event.key === ' ') {event.preventDefault(); highlight(selectedAxis === axis.key ? null : axis.key);}});
        svg.append(node); angle = next;
      });
      text(svg, 110, 110, fmt(total), 'donut-count', 'middle').id = 'donut-count';
      text(svg, 110, 131, metric, 'donut-label', 'middle').id = 'donut-label';
      document.querySelector('#dataset-donut').replaceChildren(svg);
      const legend = document.querySelector('#dataset-legend'); legend.replaceChildren();
      axes.forEach(axis => {
        const button = document.createElement('button');
        button.type = 'button'; button.dataset.axis = axis.key;
        button.style.setProperty('--chart-color', axis.color);
        button.setAttribute('aria-pressed', 'false');
        const dot = document.createElement('span'); dot.setAttribute('aria-hidden', 'true');
        const label = document.createElement('span'); label.textContent = axis.label;
        const value = document.createElement('strong'); value.textContent = `${(100 * axis[metric] / total).toFixed(1)}%`;
        button.append(dot, label, value);
        button.addEventListener('click', () => highlight(selectedAxis === axis.key ? null : axis.key));
        legend.append(button);
      });
      highlight(selectedAxis);
    }
    function highlight(key) {
      selectedAxis = key;
      const axis = axisMap[key];
      document.querySelectorAll('.donut-segment').forEach(node => {node.style.opacity = key && node.dataset.axis !== key ? '.22' : '1';});
      document.querySelectorAll('#dataset-legend button').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.axis === key)));
      document.querySelector('#donut-count').textContent = fmt(axis ? axis[metric] : data.totals[metric]);
      document.querySelector('#donut-label').textContent = axis ? axis.short : metric;
      compositionReadout.textContent = axis
        ? `${axis.label}: ${fmt(axis[metric])} ${metric} · ${(100 * axis[metric] / data.totals[metric]).toFixed(1)}% of the total.`
        : `${fmt(data.totals[metric])} ${metric} across five capability axes. Select an axis to explore.`;
    }
    const bars = document.querySelector('#dataset-bars');
    const maxHours = Math.max(...axes.map(axis => axis.hours));
    axes.forEach(axis => {
      const button = document.createElement('button'); button.type = 'button'; button.className = 'dataset-bar';
      button.style.setProperty('--chart-color', axis.color); button.style.setProperty('--bar-size', `${axis.hours / maxHours * 100}%`);
      const label = document.createElement('span'); label.textContent = axis.label;
      const value = document.createElement('strong'); value.textContent = `${axis.hours.toFixed(2)} h`;
      const track = document.createElement('span'); track.className = 'dataset-bar-track'; track.setAttribute('aria-hidden', 'true');
      const fill = document.createElement('span'); fill.className = 'dataset-bar-fill'; track.append(fill);
      button.append(label, value, track);
      const detail = `${axis.label}: ${fmt(axis.frames)} frames · ${axis.hours.toFixed(2)} h · ${(100 * axis.frames / data.totals.frames).toFixed(1)}% of recorded time.`;
      button.setAttribute('aria-label', detail);
      const show = () => {document.querySelector('#time-readout').textContent = detail;};
      button.addEventListener('pointerenter', show); button.addEventListener('focus', show); button.addEventListener('click', show);
      bars.append(button);
    });
    document.querySelectorAll('[data-dataset-metric]').forEach(button => button.addEventListener('click', () => {
      metric = button.dataset.datasetMetric;
      document.querySelectorAll('[data-dataset-metric]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
      renderComposition();
    }));
    renderComposition(); renderFlow();
    let lastWidth = flowHost.clientWidth;
    new ResizeObserver(() => {const width = flowHost.clientWidth; if (width !== lastWidth) {lastWidth = width; renderFlow();}}).observe(flowHost);
    root.setAttribute('aria-busy', 'false');
  }).catch(error => {
    root.setAttribute('aria-busy', 'false');
    const message = document.createElement('p'); message.className = 'dataset-error';
    message.textContent = 'Charts could not be loaded. Download the per-task statistics below to explore the dataset.';
    root.replaceChildren(message);
    console.error(error);
  });
})();
