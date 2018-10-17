d3.select('body')
  .append('svg').attr('id', 'svg-filters')
    .call(goo_filter);



function goo_filter(svg) {
  var defs = svg.select('defs');
  if(defs.empty()) {
    defs = svg.append('defs');
  }
  var filter = defs.append('filter').attr('id', 'goo');
  
  filter.append('feGaussianBlur')
    .attr('stdDeviation', 10)
    .attr('result', "blur")
  filter.append('feColorMatrix')
    .attr('in', 'blur')
    .attr('mode', 'matrix')
    .attr('values', '1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 35 -10')
    .attr('result', 'goo')
  filter.append('feBlend')
    .attr('in', 'SourceGraphic')
    .attr('in2', 'goo')
    .attr('operator', 'atop')
}



function bubble_loader(el, o) {
  if(o === false) {
    var loader = el.select('.bubble-loader');
      loader.selectAll('.dot').transition().duration(800).style('transform', 'scale(0)')
      loader.transition().delay(700).duration(300).style('opacity', '0').remove()
    return;
  }

  o = Object.assign({
    n: 5,
    t: 0.7,
  }, o)
  o.f = o.f || o.n;

  var els = el.append('div').attr('class', 'bubble-loader')
    .style('width', o.n/2 + 'em')
    .append('div').attr('class', 'goo')
    .selectAll('div').data(new Array(o.n));

  els.enter().append('div').attr('class', 'dot')
    .style('animation', (d, i) =>
    'bubble-loader ' + o.t + 's ease-in-out ' + ((i - o.n) * o.t / o.f) + 's infinite')
  
  els.exit().transition().duration(1).style('opacity', 0).remove();
  
}


