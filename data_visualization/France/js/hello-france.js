window.onload = function(){
  var w = 600;
  var h = 600;
  var dataset = [];

  //Create SVG element
  var svg_map = d3.select("body")
                  .append("svg")
                  .attr("width", w)
                  .attr("height", h);

  var svg_density = d3.select("body")
                      .append("svg")
                      .attr("width", w)
                      .attr("height", h);

  var svg_population = d3.select("body")
                          .append("svg")
                          .attr("width", w)
                          .attr("height", h);

  function draw_map() {
    x = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.longitude; }))
          .range([0, w]);
    y = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.latitude; }))
          .range([h, 0]);
    svg_map.selectAll('rect')
       .data(dataset)
       .enter()
       .append('rect')
       .attr('width', 1)
       .attr('height', 1)
       .attr('x', function(d) { return x(d.longitude); })
       .attr('y', function(d) { return y(d.latitude); });
  }

  function draw_density() {
    x = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.longitude; }))
          .range([0, w]);
    y = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.latitude; }))
          .range([h, 0]);
    colors = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.density; }))
          .range(['blue', 'red'])
          .interpolate(d3.interpolateRgb);

    svg_density.selectAll('rect')
       .data(dataset)
       .enter()
       .append('rect')
       .attr('width', 1)
       .attr('height', 1)
       .attr('x', function(d) { return x(d.longitude); })
       .attr('y', function(d) { return y(d.latitude); })
       .style('fill', function(d) { return colors(d.density) });
  }

  function draw_population() {
    x = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.longitude; }))
          .range([0, w]);
    y = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.latitude; }))
          .range([h, 0]);
    colors = d3.scaleLinear()
          .domain(d3.extent(dataset, function(row) { return row.population; }))
          .range(['blue', 'red'])
          .interpolate(d3.interpolateRgb);
    svg_population.selectAll('rect')
       .data(dataset)
       .enter()
       .append('rect')
       .attr('width', 1)
       .attr('height', 1)
       .attr('x', function(d) { return x(d.longitude); })
       .attr('y', function(d) { return y(d.latitude); })
       .style('fill', function(d) { return colors(d.population) });
  }

  d3.tsv('data/france.tsv')
    .row(function(d, i) {
      return {
        postalCode: +d['Postal Code'],
        inseeCode: +d.inseecode,
        place: d.place,
        longitude: +d.x,
        latitude: +d.y,
        population: +d.population,
        density: +d.density
      }
    })
    .get(function(error, rows) {
      dataset = rows;
      draw_map();
      draw_density();
      draw_population();
    });

};