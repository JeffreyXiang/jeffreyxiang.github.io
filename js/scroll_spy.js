$(document).ready(function () {
  var tocLinks = $(".toc a");

  var sections = tocLinks.map(function () {
    var target = $($(this).attr("href"));
    if (target.length) return target;
  });

  $(window).on("scroll", function () {
    var scrollTop = $(this).scrollTop();

    var current = sections.map(function () {
      if ($(this).offset().top - 100 <= scrollTop) {
        return this;
      }
    });

    current = current[current.length - 1];
    var id = current && current.length ? current[0].id : "";

    tocLinks.removeClass("active");
    if (id) {
      $(".toc a[href='#" + id + "']").addClass("active");
    }
  });
});
