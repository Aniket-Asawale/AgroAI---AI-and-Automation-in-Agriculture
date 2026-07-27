// ─────────────────────────────────────────────────────────────
// AgroAI — main.js
// Theme toggle, nav scroll, stat counters, GSAP scroll animations,
// mobile menu. No Three.js dependency.
// ─────────────────────────────────────────────────────────────
(function () {
  'use strict';

  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ─── 1. THEME TOGGLE ───────────────────────────────────────
  const THEME_KEY = 'agroai-theme';
  const themeBtn = document.getElementById('theme-toggle');

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { }
  }

  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const cur = document.documentElement.getAttribute('data-theme') || 'light';
      applyTheme(cur === 'dark' ? 'light' : 'dark');
    });
  }

  // ─── 2. NAV SCROLL ─────────────────────────────────────────
  const nav = document.getElementById('nav');
  if (nav) {
    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 60);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
  }

  // ─── 3. MOBILE MENU ────────────────────────────────────────
  const mobileBtn = document.getElementById('mobile-menu-btn');
  const navLinks = document.querySelector('.nav-links');
  if (mobileBtn && navLinks) {
    mobileBtn.addEventListener('click', () => {
      const open = navLinks.style.display === 'flex';
      navLinks.style.cssText = open
        ? ''
        : 'display:flex;flex-direction:column;gap:4px;position:absolute;top:68px;left:0;right:0;background:var(--nav-bg-scrolled);backdrop-filter:blur(20px);padding:16px 28px;border-bottom:1px solid var(--border);z-index:999;';
    });
    // Close on nav link click
    navLinks.querySelectorAll('a').forEach(a => {
      a.addEventListener('click', () => { navLinks.style.cssText = ''; });
    });
  }

  // ─── 4. SMOOTH ANCHOR SCROLL ───────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener('click', e => {
      const target = document.querySelector(a.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  // ─── 5. STAT COUNTERS ──────────────────────────────────────
  function animateCounter(el) {
    const target = parseInt(el.dataset.count, 10);
    const suffix = el.dataset.suffix || '';
    if (isNaN(target)) return;
    if (reduceMotion) { el.textContent = target + suffix; return; }

    const duration = 1400;
    const start = performance.now();
    const ease = t => 1 - Math.pow(1 - t, 3);

    (function tick(now) {
      const p = Math.min((now - start) / duration, 1);
      el.textContent = Math.round(ease(p) * target) + suffix;
      if (p < 1) requestAnimationFrame(tick);
    })(start);
  }

  if ('IntersectionObserver' in window) {
    const obs = new IntersectionObserver(entries => {
      entries.forEach(e => {
        if (e.isIntersecting) { animateCounter(e.target); obs.unobserve(e.target); }
      });
    }, { threshold: 0.5 });
    document.querySelectorAll('[data-count]').forEach(el => obs.observe(el));
  } else {
    document.querySelectorAll('[data-count]').forEach(animateCounter);
  }

  // ─── 6. GSAP SCROLL ANIMATIONS ─────────────────────────────
  function scrollSt(trigger, start) {
    return { trigger, start, once: true, toggleActions: 'play none none none' };
  }

  function refreshScrollTriggers() {
    if (window.ScrollTrigger) window.ScrollTrigger.refresh();
  }

  function ensureVisibleAnimatedElements() {
    if (!window.gsap) return;
    const { gsap } = window;
    const selectors = [
      '[data-stagger] > *',
      '.reveal-up',
      '.section-header-center',
      '.section-header-left',
      '.pipeline-step',
      '.faq-item',
      '.cta-band',
      '.trust-item',
    ].join(', ');
    document.querySelectorAll(selectors).forEach(el => {
      const opacity = parseFloat(getComputedStyle(el).opacity);
      if (opacity < 0.95) {
        gsap.set(el, { opacity: 1, y: 0, x: 0, scale: 1, clearProps: 'transform' });
      }
    });
  }

  function initGSAP() {
    if (reduceMotion || !window.gsap || !window.ScrollTrigger) return;

    const { gsap } = window;
    gsap.registerPlugin(window.ScrollTrigger);
    document.documentElement.classList.add('js-anim');

    // Hero — staggered entrance (page load, not scroll-triggered)
    const heroTl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    heroTl
      .from('.hero-badge', { y: 20, opacity: 0, duration: 0.6 }, 0.1)
      .from('.hero h1', { y: 36, opacity: 0, duration: 0.7 }, 0.2)
      .from('.hero-subtitle', { y: 24, opacity: 0, duration: 0.6 }, 0.35)
      .from('.hero-actions .btn', { y: 18, opacity: 0, stagger: 0.1, duration: 0.55 }, 0.5)
      .from('.hero-stats .hero-stat', { y: 14, opacity: 0, stagger: 0.08, duration: 0.5 }, 0.65)
      .from('.hero-right', { x: 40, opacity: 0, duration: 0.85, ease: 'expo.out' }, 0.25);

    // Reveal up — generic scroll-triggered fade+slide
    gsap.utils.toArray('.reveal-up').forEach(el => {
      gsap.fromTo(el,
        { y: 40, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.75, ease: 'power3.out',
          immediateRender: false,
          scrollTrigger: scrollSt(el, 'top 85%'),
        }
      );
    });

    // Staggered card grids
    document.querySelectorAll('[data-stagger]').forEach(grid => {
      const items = Array.from(grid.children);
      gsap.fromTo(items,
        { y: 44, opacity: 0 },
        {
          y: 0, opacity: 1,
          duration: 0.65, ease: 'power2.out',
          stagger: { each: 0.08, from: 'start' },
          immediateRender: false,
          scrollTrigger: scrollSt(grid, 'top 82%'),
        }
      );
    });

    // Section headers
    gsap.utils.toArray('.section-header-center, .section-header-left').forEach(el => {
      gsap.fromTo(el,
        { y: 30, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.7, ease: 'power3.out',
          immediateRender: false,
          scrollTrigger: scrollSt(el, 'top 86%'),
        }
      );
    });

    // Pipeline steps
    gsap.utils.toArray('.pipeline-step').forEach((el, i) => {
      gsap.fromTo(el,
        { x: -24, opacity: 0 },
        {
          x: 0, opacity: 1, duration: 0.55, ease: 'power2.out',
          delay: i * 0.06,
          immediateRender: false,
          scrollTrigger: scrollSt(el, 'top 88%'),
        }
      );
    });

    // FAQ items
    gsap.utils.toArray('.faq-item').forEach((el, i) => {
      gsap.fromTo(el,
        { y: 20, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.45, ease: 'power2.out',
          delay: i * 0.05,
          immediateRender: false,
          scrollTrigger: scrollSt(el, 'top 90%'),
        }
      );
    });

    // CTA band
    const ctaBand = document.querySelector('.cta-band');
    if (ctaBand) {
      gsap.fromTo(ctaBand,
        { y: 32, opacity: 0, scale: 0.98 },
        {
          y: 0, opacity: 1, scale: 1, duration: 0.85, ease: 'power3.out',
          immediateRender: false,
          scrollTrigger: scrollSt(ctaBand, 'top 84%'),
        }
      );
    }

    // Trust band items
    gsap.utils.toArray('.trust-item').forEach((el, i) => {
      gsap.fromTo(el,
        { y: 16, opacity: 0 },
        {
          y: 0, opacity: 1, duration: 0.45, ease: 'power2.out',
          delay: i * 0.07,
          immediateRender: false,
          scrollTrigger: scrollSt(el, 'top 90%'),
        }
      );
    });

    refreshScrollTriggers();
    window.addEventListener('load', () => {
      refreshScrollTriggers();
      ensureVisibleAnimatedElements();
    });
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(refreshScrollTriggers);
    }
  }

  function waitForGSAP(cb, attempts = 50) {
    if (window.gsap && window.ScrollTrigger) return cb();
    if (attempts <= 0) return;
    setTimeout(() => waitForGSAP(cb, attempts - 1), 50);
  }

  function bootGSAP() {
    waitForGSAP(initGSAP);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', bootGSAP);
  } else {
    bootGSAP();
  }

  // ─── 7. HERO DASH CARD — live-looking sensor updates ───────
  const sensorValues = [
    { n: 72, p: 45, k: 88, ph: 6.8, crop: '🌾 Wheat', conf: 89, confVal: '89.3%' },
    { n: 58, p: 62, k: 95, ph: 7.1, crop: '🌽 Maize', conf: 76, confVal: '76.2%' },
    { n: 90, p: 48, k: 70, ph: 6.5, crop: '🍚 Rice', conf: 84, confVal: '84.1%' },
    { n: 45, p: 30, k: 60, ph: 7.4, crop: '🌱 Chickpea', conf: 81, confVal: '81.9%' },
  ];
  let svIdx = 0;

  function rotateSensorDisplay() {
    const card = document.getElementById('hero-dash-card');
    if (!card) return;
    const chips = card.querySelectorAll('.sensor-chip');
    const confLabel = card.querySelector('.conf-label');
    const confVal = card.querySelector('.conf-val');
    const confFill = card.querySelector('.conf-bar-fill');

    svIdx = (svIdx + 1) % sensorValues.length;
    const sv = sensorValues[svIdx];

    if (chips[0]) chips[0].textContent = `N: ${sv.n} ppm`;
    if (chips[1]) chips[1].textContent = `P: ${sv.p} ppm`;
    if (chips[2]) chips[2].textContent = `K: ${sv.k} ppm`;
    if (chips[3]) chips[3].textContent = `pH: ${sv.ph}`;
    if (confLabel) confLabel.textContent = sv.crop;
    if (confVal) confVal.textContent = sv.confVal;
    if (confFill) confFill.style.width = sv.conf + '%';
  }

  // Rotate every 3 seconds to look "live"
  if (!reduceMotion) {
    setInterval(rotateSensorDisplay, 3000);
  }

})();
