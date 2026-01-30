(() => {
  const scrollSidebarToCurrent = () => {
    const sidebar = document.querySelector(".sidebar-scroll");
    if (!sidebar || sidebar.scrollHeight <= sidebar.clientHeight) {
      return;
    }

    const currentLink =
      sidebar.querySelector(".sidebar-tree li.current-page > .reference") ||
      sidebar.querySelector(".sidebar-tree li.current > .reference");

    if (!currentLink) {
      return;
    }

    const sidebarRect = sidebar.getBoundingClientRect();
    const linkRect = currentLink.getBoundingClientRect();
    const isAbove = linkRect.top < sidebarRect.top;
    const isBelow = linkRect.bottom > sidebarRect.bottom;

    if (!isAbove && !isBelow) {
      return;
    }

    const offset = linkRect.top - sidebarRect.top;
    const target = offset - sidebar.clientHeight * 0.35;
    sidebar.scrollTop += target;
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", scrollSidebarToCurrent);
  } else {
    scrollSidebarToCurrent();
  }
})();
