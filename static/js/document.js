(() => {
  const toast = document.querySelector('#doc-toast');
  let timer;
  document.querySelectorAll('.doc-section table').forEach(table => {
    const wrap = document.createElement('div');
    wrap.className = 'doc-table-wrap';
    wrap.tabIndex = 0;
    wrap.setAttribute('role', 'region');
    wrap.setAttribute('aria-label', 'Scrollable documentation table');
    table.before(wrap);
    wrap.append(table);
  });
  document.querySelectorAll('.doc-section pre').forEach(pre => {
    const code = pre.querySelector('code');
    if (!code) return;
    const label = document.createElement('span');
    label.className = 'doc-code-label';
    label.textContent = [...code.classList].find(c => c.startsWith('language-'))?.slice(9) || 'text';
    const copy = document.createElement('button');
    copy.type = 'button';
    copy.className = 'doc-copy';
    copy.textContent = 'Copy';
    copy.setAttribute('aria-label', 'Copy code');
    copy.addEventListener('click', async () => {
      const text = code.textContent;
      try {
        if (navigator.clipboard && window.isSecureContext) await navigator.clipboard.writeText(text);
        else {
          const field = document.createElement('textarea');
          field.value = text;
          field.style.cssText = 'position:fixed;left:-9999px;top:0';
          document.body.append(field);
          field.select();
          const copied = document.execCommand('copy');
          field.remove();
          if (!copied) throw new Error('Copy failed');
        }
        toast.textContent = 'Code copied';
      } catch {
        toast.textContent = 'Select the code to copy it';
      }
      toast.classList.add('visible');
      clearTimeout(timer);
      timer = setTimeout(() => toast.classList.remove('visible'), 1800);
    });
    pre.append(label, copy);
  });
  const links = [...document.querySelectorAll('.doc-sidebar nav a')];
  const sections = links.map(a => document.querySelector(a.hash));
  const update = () => {
    let current = sections[0];
    for (const section of sections) if (section.getBoundingClientRect().top <= 130) current = section;
    links.forEach(link => {
      const active = link.hash === '#' + current.id;
      link.classList.toggle('active', active);
      if (active) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  };
  window.addEventListener('scroll', update, {passive: true});
  update();
})();
