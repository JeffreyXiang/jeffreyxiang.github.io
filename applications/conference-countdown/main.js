function loadDates() {
    $.get("/applications/conference-countdown/dates.json", function(data){
        buildList(data);
    });
}

function colorBar(f) {
    return 'rgb(' + parseInt(224 * Math.min(1, 2 * f)) + ',' + parseInt(224 * Math.min(1, 2 - 2 * f)) + ',0)';
}

function formatWidth2(n) {
    return ("00" + n).slice(-2);
}

function buildList(data) {
    let card = $(".conference_card").get(0).outerHTML;
    let remainPhase = $(".conference_time_remain_phase").get(0).outerHTML;
    let progressPhase = $(".conference_time_progress_phase").get(0).outerHTML;
    let absPhase = $(".conference_time_absolute_phase").get(0).outerHTML;
    let finalHTML = "";
    let today = new Date();
    for (let i = data.length - 1; i >= 0; i--) {
        let cardHTML = card
            .replace("{ConferenceName}", data[i]["name"].slice(0, -4) + "<br>" + data[i]["name"].slice(-4))
            .replace("{link}", data[i]["link"]);
        let remainHTML = "";
        let progressHTML = "";
        let absHTML = "";
        let progress;
        let prevDate;
        for (let j = 0; j < data[i]["dates"].length; j++) {
            let date = new Date(data[i]["dates"][j]['date']);
            if (j == 0) {
                prevDate = date - 3600000 * 24 * 100;
                if (today < prevDate) progress = 0;
            }
            if (j == data[i]["dates"].length - 1) {
                if (today > date) progress = 1;
            }
            if (today > prevDate && today < date) progress = (j + (today - prevDate) / (date - prevDate)) / data[i]["dates"].length;
            remainHTML += remainPhase
                .replace("{Weeks}", parseInt(Math.max(0, (date-today)/3600000/24/7)))
                .replace("{Days}", parseInt(Math.max(0, (date-today)/3600000/24%7)))
                .replace("{Hours}", parseInt(Math.max(0, (date-today)/3600000%24)));
            progressHTML += progressPhase.replace("{color}", colorBar(j / (data[i]["dates"].length - 1)));
            absHTML += absPhase
                .replace("{PhaseAbsTime}", date.getFullYear() + "/" + (date.getMonth() + 1) + "/" + date.getDate() + " " + formatWidth2(date.getHours()) + ":" + formatWidth2(date.getMinutes()))
                .replace("{PhaseName}", data[i]["dates"][j]['phasename']);
            prevDate = date;
        }
        finalHTML += cardHTML
            .replace("{progress}", 100 * progress + "%")
            .replace(remainPhase, remainHTML)
            .replace(progressPhase, progressHTML)
            .replace(absPhase, absHTML);
    }
    $(".conference_list").html(finalHTML);
}

$(function(){
    loadDates();
});
