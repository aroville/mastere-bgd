var w = 800, h = 800;
var dataset = [];
var sizes, colors;
var placeLabel,
    postalCodeLabel,
    populationLabel,
    densityLabel;


//Create SVG element
var svg = d3.select("body")
            .append("svg")
            .attr("width", w)
            .attr("height", h)
            .style("margin-left", '10px')
            .style("margin-top", '10px');


function log(v) { return Math.log(1 + v); }
function size(d) { return sizes(log(d.population)); }
function color(d) {
  return d3.rgb(255, 255-colors(log(d.density)), 0, 0.8).toString();
}


function scale(attr, range, linearize=false) {
  return d3.scaleLinear()
           .domain(d3.extent(dataset, function(d) {
             return linearize ? log(d[attr]) : d[attr];
           })).range(range);
}


function drawCross(x, y) {
  var lh = document.getElementById('lh');
  if (lh) lh.outerHTML = '';
  var lv = document.getElementById('lv');
  if (lv) lv.outerHTML = '';

  svg.append('line')
    .attr('id', 'lh')
    .attr('x1', x-14)
    .attr('y1', y)
    .attr('x2', x+14)
    .attr('y2', y)
    .style('stroke', 'rgb(0,255,0)')
    .style('stroke-width', 2);

  svg.append('line')
    .attr('id', 'lv')
    .attr('x1', x)
    .attr('y1', y-14)
    .attr('x2', x)
    .attr('y2', y+14)
    .style('stroke', 'rgb(0,255,0)')
    .style('stroke-width', 2);
}


function updateLabel(d) {
  placeLabel.innerHTML = 'Ville: ' + d.place;
  postalCodeLabel.innerHTML = 'Code postal: ' + d.postalCode;
  populationLabel.innerHTML = 'Population: ' + d.population;
  densityLabel.innerHTML = 'Densité: ' + d.density;
  drawCross(Math.round(x(d.longitude)), Math.round(y(d.latitude)));
}


function hideLabel(d) {
  placeLabel.innerHTML = '';
  postalCodeLabel.innerHTML = '';
  populationLabel.innerHTML = '';
  densityLabel.innerHTML = '';
}


function createLabels() {
  var container = document.createElement('div');
  container.setAttribute('id', 'label-div')

  placeLabel = document.createElement('p');
  placeLabel.setAttribute('id', 'place-label');
  container.appendChild(placeLabel);

  postalCodeLabel = document.createElement('p');
  postalCodeLabel.setAttribute('id', 'postalCode-label');
  container.appendChild(postalCodeLabel);

  populationLabel = document.createElement('p');
  populationLabel.setAttribute('id', 'population-label');
  container.appendChild(populationLabel);

  densityLabel = document.createElement('p');
  densityLabel.setAttribute('id', 'density-label');
  container.appendChild(densityLabel);

  document.getElementsByTagName('body')[0]
    .appendChild(container);
}


function draw() {
  x = scale('longitude', [0, w]);
  y = scale('latitude', [h, 0]);
  colors = scale('density', [0, 255], true);
  sizes = scale('population', [0.6, 3.], true);

  createLabels();
  svg.selectAll('ellipse')
     .data(dataset).enter()
     .append('ellipse')
     .attr('rx', size)
     .attr('ry', size)
     .attr('cx', function(d) { return x(d.longitude); })
     .attr('cy', function(d) { return y(d.latitude); })
     .style('fill', color)
     .on('mouseover', updateLabel);
}


d3.tsv('data/france.tsv')
  .row(function(d, i) {
    var x = +d.x;
    var y = +d.y;
    if (isNaN(x) || isNaN(y)) return;

    return {
      postalCode: +d['Postal Code'],
      place: d.place,
      longitude: x,
      latitude: y,
      population: +d.population,
      density: +d.density
    }
  })
  .get(function(error, rows) {
    dataset = rows.sort(function(a,b) {
      return +a.density - +b.density;
    });;
    draw();
  });
