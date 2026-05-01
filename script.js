/* ─── Editorial Portfolio — Interactive JS ─── */

document.addEventListener('DOMContentLoaded', () => {

    /* ── Custom Cursor ── */
    const cursor = document.getElementById('cursor');

    if (cursor && window.matchMedia('(pointer: fine)').matches) {
        let cursorX = 0, cursorY = 0;
        let rafId;

        document.addEventListener('mousemove', (e) => {
            cursorX = e.clientX;
            cursorY = e.clientY;
            cancelAnimationFrame(rafId);
            rafId = requestAnimationFrame(() => {
                cursor.style.left = cursorX + 'px';
                cursor.style.top  = cursorY + 'px';
            });
        });

        const hoverTargets = 'a, button, [role="button"], .sg-tags span, .proj-row-main';
        document.querySelectorAll(hoverTargets).forEach(el => {
            el.addEventListener('mouseenter', () => cursor.classList.add('expanded'));
            el.addEventListener('mouseleave', () => cursor.classList.remove('expanded'));
        });

        document.addEventListener('mouseleave', () => { cursor.style.opacity = '0'; });
        document.addEventListener('mouseenter', () => { cursor.style.opacity = '1'; });
    } else if (cursor) {
        cursor.style.display = 'none';
    }

    /* ── Reading Progress Bar ── */
    const progressBar = document.getElementById('progress-bar');

    function updateProgress() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight - window.innerHeight;
        const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
        if (progressBar) progressBar.style.width = pct + '%';
    }

    window.addEventListener('scroll', updateProgress, { passive: true });

    /* ── Nav scroll state ── */
    const nav = document.getElementById('nav');

    function updateNav() {
        if (!nav) return;
        if (window.scrollY > 40) {
            nav.classList.add('scrolled');
        } else {
            nav.classList.remove('scrolled');
        }
    }

    window.addEventListener('scroll', updateNav, { passive: true });
    updateNav();

    /* ── Mobile hamburger ── */
    const hamburger = document.getElementById('hamburger');
    const navMobile = document.getElementById('nav-mobile');

    if (hamburger && navMobile) {
        hamburger.addEventListener('click', () => {
            const isOpen = navMobile.classList.toggle('open');
            hamburger.classList.toggle('open', isOpen);
            hamburger.setAttribute('aria-expanded', String(isOpen));
        });

        navMobile.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                navMobile.classList.remove('open');
                hamburger.classList.remove('open');
                hamburger.setAttribute('aria-expanded', 'false');
            });
        });
    }

    /* ── Project row accordion ── */
    document.querySelectorAll('.project-row').forEach(row => {
        const main  = row.querySelector('.project-row-main');
        const toggle = row.querySelector('.proj-toggle');
        const links  = row.querySelector('.proj-links');

        function openRow() {
            const isOpen = row.classList.toggle('open');
            if (toggle) toggle.setAttribute('aria-label', isOpen ? 'Close project details' : 'Open project details');
            if (main) main.setAttribute('aria-expanded', String(isOpen));

            // Close other open rows
            if (isOpen) {
                document.querySelectorAll('.project-row.open').forEach(other => {
                    if (other !== row) {
                        other.classList.remove('open');
                        const otherToggle = other.querySelector('.proj-toggle');
                        const otherMain   = other.querySelector('.project-row-main');
                        if (otherToggle) otherToggle.setAttribute('aria-label', 'Open project details');
                        if (otherMain)   otherMain.setAttribute('aria-expanded', 'false');
                    }
                });
            }
        }

        if (main) {
            main.addEventListener('click', (e) => {
                // Don't trigger if clicking a link inside proj-links
                if (links && links.contains(e.target)) return;
                openRow();
            });

            main.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    openRow();
                }
            });
        }
    });

    /* ── Scroll reveal ── */
    const revealEls = document.querySelectorAll('.reveal');

    if (revealEls.length > 0) {
        const revealObs = new IntersectionObserver((entries) => {
            entries.forEach((entry, i) => {
                if (entry.isIntersecting) {
                    // Stagger siblings slightly
                    const siblings = entry.target.parentElement
                        ? Array.from(entry.target.parentElement.querySelectorAll('.reveal'))
                        : [];
                    const idx = siblings.indexOf(entry.target);
                    setTimeout(() => {
                        entry.target.classList.add('visible');
                    }, Math.min(idx * 80, 320));
                    revealObs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

        revealEls.forEach(el => revealObs.observe(el));
    }

    /* ── Smooth scroll for anchor links ── */
    document.querySelectorAll('a[href^="#"]').forEach(link => {
        link.addEventListener('click', (e) => {
            const target = document.querySelector(link.getAttribute('href'));
            if (!target) return;
            e.preventDefault();
            const navHeight = nav ? nav.offsetHeight : 68;
            const top = target.getBoundingClientRect().top + window.scrollY - navHeight;
            window.scrollTo({ top, behavior: 'smooth' });
        });
    });

    /* ── Active nav link highlight ── */
    const sections  = document.querySelectorAll('section[id]');
    const navLinks  = document.querySelectorAll('.nav-link');

    if (sections.length && navLinks.length) {
        const sectionObs = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const id = entry.target.id;
                    navLinks.forEach(link => {
                        link.style.color = link.getAttribute('href') === '#' + id
                            ? 'var(--rust)' : '';
                    });
                }
            });
        }, { threshold: 0.35 });

        sections.forEach(s => sectionObs.observe(s));
    }

});
