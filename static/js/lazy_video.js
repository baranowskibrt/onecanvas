// Defer <video> downloads until the element enters the viewport, à la
// https://depth-anything-3.github.io/. Videos opt in with class="lazy-video"
// and use data-src on their <source> children (and optionally on the <video>
// itself) instead of src. The IntersectionObserver swaps data-src -> src,
// calls load(), and re-attempts autoplay.

(function () {
  function activate(video) {
    if (video.dataset.lazyLoaded === '1') return;
    video.dataset.lazyLoaded = '1';

    video.querySelectorAll('source[data-src]').forEach(function (s) {
      s.src = s.dataset.src;
      s.removeAttribute('data-src');
    });
    if (video.dataset.src && !video.getAttribute('src')) {
      video.src = video.dataset.src;
      video.removeAttribute('data-src');
    }

    video.load();

    if (video.autoplay) {
      var p = video.play();
      if (p && typeof p.catch === 'function') p.catch(function () {});
    }
  }

  function init() {
    var videos = document.querySelectorAll('video.lazy-video');
    if (!videos.length) return;

    if (!('IntersectionObserver' in window)) {
      videos.forEach(activate);
      return;
    }

    var io = new IntersectionObserver(function (entries, obs) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          activate(e.target);
          obs.unobserve(e.target);
        }
      });
    }, { rootMargin: '200px' });

    videos.forEach(function (v) { io.observe(v); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
