(() => {
const select = document.getElementById("algorithm");
const fieldsets = [...document.querySelectorAll(".algorithm-fields")];
if (!select || !fieldsets.length) {
  return;
}

function showSelected() {
  fieldsets.forEach((fieldset) => {
    const active = fieldset.dataset.algorithm === select.value;
    fieldset.hidden = !active;
    fieldset.disabled = !active;
  });
}

select.addEventListener("change", showSelected);
showSelected();
})();
