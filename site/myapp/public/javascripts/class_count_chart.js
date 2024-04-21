
let count_margin_left = 82;
let count_margin_right = 25;
let count_margin_top = 50;
let count_xScale;
let count_chart_axis;

function get_count_chart_data() {

    let count_chart_data = [0, 0, 0, 0];
    for (let i = 0; i < predictions[cur_image_name]["detector_scores"].length; i++) {
        if (predictions[cur_image_name]["detector_scores"][i] > 0.5) {
            let obj_cls = predictions[cur_image_name]["classes"][i];
            count_chart_data[obj_cls]++;
        }
    }

    return count_chart_data;

}

function create_count_chart() {

    let count_chart_data = get_count_chart_data();
    let max_count = Math.max(...count_chart_data);

    $("#count_chart").empty();

    let chart_width = $("#count_chart").width();
    let chart_height = $('#count_chart').height();



    let count_svg = d3.select("#count_chart")
        .append("svg")
        .attr("width", chart_width)
        .attr("height", chart_height);

    let chart = d3.select("#count_chart").select("svg").append("g");


    count_chart_axis = count_svg.append("g")
                    .attr("class", "x axis")
                    .attr("transform", "translate(" + count_margin_left + "," + (0.4 * count_margin_top) + ")");

    count_xScale = d3.scaleLinear()
                .domain([0, max_count])
                .range([0, chart_width - count_margin_left - count_margin_right]);

    let count_yScale = d3.scaleLinear()
                .domain([0, object_classes.length])
                .range([0.45 * count_margin_top, chart_height]);



    count_chart_axis.call(d3.axisTop(count_xScale).ticks(2).tickFormat(d3.format("d")));


    chart.selectAll(".text")
         .data(object_classes)
         .enter()
         .append("text")
         .attr("class", "chart_text")
         .attr("x", count_margin_left - 10)
         .attr("y", function(d, i) {
            return count_yScale(i) + ((count_yScale(1) - count_yScale(0)) / 2);
         })
         .attr("alignment-baseline", "middle")
         .attr("text-anchor", "end")
         .attr("font-size", "12px")
         .text(function(d) { return d; })
         .style("cursor", "default");


    chart.selectAll(".num_text")
        .data([0,1,2,3])
        .enter()
        .append("text")
        .attr("class", "num_text")
        .attr("x", function(d, i) { 
        return count_margin_left + count_xScale(count_chart_data[i]) + 5; 
        })
        .attr("y", function(d, i) {
        return count_yScale(i) + ((count_yScale(1) - count_yScale(0)) / 2);
        })
        .attr("alignment-baseline", "middle")
        .attr("text-anchor", "start")
        .attr("font-size", "12px")
        .text(function(d, i) { 
        return count_chart_data[i]; 
        })
        .style("cursor", "default");



    chart.selectAll(".bar")
        .data([0,1,2,3])
        .enter()
        .append("rect")
        .attr("class", "bar")
        .attr("id", function (d, i) { return "rect" + i; })
        .attr("x", count_margin_left)
        .attr("y", function(d, i) {
        return count_yScale(i) + 2.5;
        })
        .attr("width", function(d, i) {
        return count_xScale(count_chart_data[i]);
        })
        .attr("height", function(d, i) {
        return count_yScale(1) - count_yScale(0) - 5;
        })
        .attr("fill", function(d, i) {
        return overlay_appearance["colors"][i];
        })
        .attr("shape-rendering", "crispEdges");


}


function update_count_chart() {

    let count_chart_data = get_count_chart_data();

    let max_count = Math.max(...count_chart_data);
    count_xScale.domain([0, max_count]);
    count_chart_axis.transition().duration(250).call(d3.axisTop(count_xScale).ticks(2));

    d3.selectAll(".bar")
        .data([0,1,2,3])
        .transition()
        .duration(250)
        .attr("fill", function(d, i) {
            return overlay_appearance["colors"][i];
         })
        .attr("width", function(d, i) {
            return count_xScale(count_chart_data[i]);
        });


    d3.selectAll(".num_text")
        .data([0,1,2,3])
        .transition()
        .duration(250)
        .attr("x", function(d, i) { 
            return count_margin_left + count_xScale(count_chart_data[i]) + 5; 
        })
        .text(function(d, i) { 
            return count_chart_data[i]; 
        });

}