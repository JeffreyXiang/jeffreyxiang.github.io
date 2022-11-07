function loadDates() {
    $.get("./dates.json", function(data){
        alert("Data: " + data);
    });
}

$(function(){
    loadDates();
});
