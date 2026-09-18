// ProyecKeras Landing Page - Interactive Script

document.addEventListener('DOMContentLoaded', () => {
    // 1. Interactive Asset Selector in Preview Mockup
    const assetItems = document.querySelectorAll('.sidebar-mock .sidebar-item');
    const metricR2 = document.querySelector('.metric-pill:nth-child(1) .val');
    const metricMAE = document.querySelector('.metric-pill:nth-child(2) .val');
    const metricRMSE = document.querySelector('.metric-pill:nth-child(3) .val');
    const metricRSI = document.querySelector('.metric-pill:nth-child(4) .val');
    const chartBars = document.querySelectorAll('.chart-bars .bar');

    const assetData = {
        'IBM': { r2: '0.942', mae: '$1.42', rmse: '$1.86', rsi: '58.4', heights: [45, 52, 60, 58, 68, 75, 72, 84, 90, 95] },
        'AAPL': { r2: '0.965', mae: '$1.15', rmse: '$1.45', rsi: '64.2', heights: [50, 55, 62, 70, 78, 80, 85, 88, 92, 98] },
        'WTI': { r2: '0.898', mae: '$2.10', rmse: '$2.75', rsi: '43.1', heights: [70, 65, 60, 55, 52, 48, 50, 54, 58, 62] },
        'GOLD': { r2: '0.978', mae: '$8.50', rmse: '$11.20', rsi: '69.8', heights: [60, 65, 72, 78, 82, 85, 90, 93, 96, 100] }
    };

    assetItems.forEach(item => {
        item.addEventListener('click', () => {
            // Find parent list section
            const parent = item.parentElement;
            parent.querySelectorAll('.sidebar-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');

            // Update stats if it's an asset
            const text = item.textContent;
            let key = null;
            if (text.includes('IBM')) key = 'IBM';
            else if (text.includes('AAPL')) key = 'AAPL';
            else if (text.includes('WTI')) key = 'WTI';
            else if (text.includes('Oro') || text.includes('GOLD')) key = 'GOLD';

            if (key && assetData[key]) {
                const d = assetData[key];
                if (metricR2) metricR2.textContent = d.r2;
                if (metricMAE) metricMAE.textContent = d.mae;
                if (metricRMSE) metricRMSE.textContent = d.rmse;
                if (metricRSI) metricRSI.textContent = d.rsi;

                // Animate bars
                chartBars.forEach((bar, idx) => {
                    if (d.heights[idx] !== undefined) {
                        bar.style.height = `${d.heights[idx]}%`;
                    }
                });
            }
        });
    });

    // 2. Interactive Terminal Copy Button Feedback
    const copyBtns = document.querySelectorAll('.btn-copy');
    copyBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const originalText = btn.textContent;
            btn.textContent = '¡Copiado! ✓';
            btn.style.background = 'var(--accent-emerald)';
            btn.style.color = '#fff';
            setTimeout(() => {
                btn.textContent = originalText;
                btn.style.background = '';
                btn.style.color = '';
            }, 2000);
        });
    });

    // 3. Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // 4. Counter Animation for Stat Cards
    const statValues = document.querySelectorAll('.stat-value');
    let animated = false;

    const animateStats = () => {
        if (animated) return;
        statValues.forEach(stat => {
            const text = stat.textContent.trim();
            if (text.includes('+')) {
                const val = parseInt(text);
                let curr = 0;
                const timer = setInterval(() => {
                    curr++;
                    stat.textContent = `${curr}+`;
                    if (curr >= val) clearInterval(timer);
                }, 100);
            }
        });
        animated = true;
    };

    window.addEventListener('scroll', () => {
        const statsSection = document.querySelector('.stats-grid');
        if (statsSection) {
            const rect = statsSection.getBoundingClientRect();
            if (rect.top <= window.innerHeight && rect.bottom >= 0) {
                animateStats();
            }
        }
    });
});
