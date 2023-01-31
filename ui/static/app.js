function add_item_to_history(item){
    $('#history').append('<li class=list-group-item>' + item + "</li>");
}

function update_recommendations() {
    var last_movies = []
    var history_elems = $('#history>li');
    for (var i = 0; i < history_elems.length; ++i){
        last_movies.push(history_elems[i].textContent);
    }
    $.post("/recommend", {"history": last_movies}).done(function(response){
        $("#recommendations").empty();
        for (var i = 0; i < response.length; ++i){
            $("#recommendations").append('<li class=list-group-item>' + response[i] + "</li>");
        }
    });
}

$(document).ready(function() {
    // Defining the local dataset
    var movies = new Bloodhound({
        datumTokenizer: Bloodhound.tokenizers.whitespace,
        queryTokenizer: Bloodhound.tokenizers.whitespace,
        remote: {
            url: '/search?keyword=%QUERY',
            wildcard: '%QUERY'
        }
    });

    // Initializing the typeahead
    $('#search').typeahead({
        hint: true,
        highlight: true, /* Enable substring highlighting */
        minLength: 1 /* Specify minimum characters required for showing suggestions */
    },
    {
        name: 'moviesSearch',
        source: movies
    });


    $('#search').bind('typeahead:selected', function(obj, datum, name) {
        add_item_to_history(datum);
        $('#search').typeahead('val', '');
        update_recommendations();
    });
    // all custom jQuery will go here
});