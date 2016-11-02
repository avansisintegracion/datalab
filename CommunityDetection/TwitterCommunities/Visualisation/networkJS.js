

    function loadInfo(data, index) {

        $.getJSON("Info4.json", function(info) {

            var influencers = "";
            var hashtags = "";
            var tweets = "";

            var k = data.nodes[index].group;
            for (i = 0; i < info['influence'][k].length; i++) {
                influencers = influencers + '<br>' + info['influence'][k][i];

            }
            for (i = 0; i < info['hashtags'][k].length; i++) {
                hashtags = hashtags + '<br>' + info['hashtags'][k][i][0] + " " + info['hashtags'][k][i][1] + "#";
            }
            for (i = 0; i < info['tweets'][k].length; i++) {
                tweets = tweets + '<br>' + '<br>' + info['tweets'][k][i][0] + " " + info['tweets'][k][i][1] + " RT";
            }

            var html="<table><tr><td width=25%><b>Users </b><br>" +
                      influencers+"</td><td width=20%><b>Hashtags </b><br>"+hashtags+
                      "</td><td><b>Tweets </b><br>"+tweets+"</td></tr></table>"

            $(".modal-title").html("Influencing this community");
            $(".modal-body").html(html);
            $("#myModal").modal("show");



        });

    }





    function redrawAll() {

        var container = document.getElementById('mynetwork');
        var options = {
            nodes: {
                shape: 'dot',
                scaling: {
                    min: 10,
                    max: 30
                },
                font: {
                    size: 8,
                    face: 'Tahoma'
                }
            },
            edges: {
                color: {
                    inherit: true
                },
                width: 0.15,
                smooth: {
                    type: 'continuous'
                }
            },
            interaction: {
                hideEdgesOnDrag: true,
                tooltipDelay: 200
            },
            physics: false,
            interaction: {
                hover: true
            }
        };

        $.getJSON("twitterSmallInfoMap4.json", function(data) {
            var network = new vis.Network(container, data, options);
            network.on("click", function(params) {


                params.event = "[original event]";
                var index = _.findIndex(data.nodes, function(o) {
                    return o.id == params.nodes[0];
                });
                if (index > 0) {

                    loadInfo(data, index);
                    network.setOptions(options);
                }
            });
        });


    }
