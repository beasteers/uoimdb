// hash: filter|time|row size

// create image grid container
var row = d3.select('#grid')
	.append('div').attr('class', 'container')
	.append('div').attr('class', 'row');

var timeline = d3.select('#timeline');

// bind control events
var controls = d3.select('#controls');

controls.select('#rowCount').attr('value', get_hash()[2] || null) // change number of columns
	.attr('max', config.GRID.SIZES.length-1)
	.on('input', changeRowCount);


function drawTimeline(data, prev_day, next_day) {
	
	// timeline.append('a').attr('class', 'timeblock timebutton')
	// 	.html('&larr; Cal')
	// 	.attr('href', BASE_URL + 'cal');

	// if(prev_day){
	// 	timeline.append('a').attr('class', 'timeblock timebutton')
	// 		.html('&larr; ' + prev_day)
	// 		.attr('href', BASE_URL + 'cal/' + prev_day);
	// }
	

	var blocks = timeline.selectAll('.timeblock:not(.timebutton)')
		.data(data).enter()
		.append('div').attr('class', 'timeblock');

	// if(next_day){
	// 	timeline.append('a').attr('class', 'timeblock timebutton')
	// 		.html('&rarr; ' + next_day)
	// 		.attr('href', BASE_URL + 'cal/' + next_day);
	// }

	// tooltip
	blocks.on('mouseover', function(d){
			tooltip.classed('visible', true)
				.style('top', d3.event.pageY+'px')
				.style('left', d3.event.pageX+'px')
				.html('');
			tooltip.append('span').call(badge, 'dark').text(d.image_count + ' images');
			tooltip.append('span').call(badge, 'dark').text(d.label_count + ' labels');
		})
		.on('mousemove', function(d){
			tooltip.style('top', d3.event.pageY+'px').style('left', d3.event.pageX+'px')
		})
		.on('mouseout', function(d){
			tooltip.classed('visible', false)
		})
		.on('click', function(d){
			d3.select(this.parentNode).selectAll('.timeblock')
				.classed('selected', false);
			d3.select(this).classed('selected', true);

			set_hash(d.date, 1);

			preloadImages(d.srcs);
			d3.json(BASE_URL + 'select-images?srcs=' + JSON.stringify(d.srcs), function(data){
				console.log(data);
				updateImages(data);
				// add on column sizes
				changeRowCount.apply(controls.select('#rowCount').node());
			})
			
		})

	// fill remaining space
	timeline.append('div').attr('class', 'timefiller');
	// add badge labels
	blocks.append('span').call(badge, 'dark').text((d) => d.date + ' ' + d.label_count + '/' + d.image_count);

	// select a timeblock. either the first one, or from the url hash
	var selected = get_hash()[1] || data[0].date;
	blocks.filter((d) => d.date == selected).each(function(d, i) {
    	d3.select(this).classed('selected', true).on("click").apply(this, [d, i]); 
    });
}


function changeRowCount(){
	var i = parseInt(this.value);
	row.selectAll('.image-cell')
		.attr('class', 'image-cell'
			 + ' col-lg-' + config.GRID.SIZES[i]
			 + ' col-md-' + config.GRID.SIZES[Math.min(i+1, config.GRID.SIZES.length-1)]
			 + ' col-sm-' + config.GRID.SIZES[Math.min(i+2, config.GRID.SIZES.length-1)]
			 + ' col-'    + config.GRID.SIZES[Math.min(i+3, config.GRID.SIZES.length-1)]);
	set_hash(i, 2);
}

