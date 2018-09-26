var gridSizes = [1, 2, 3, 4, 6, 12];

var container = d3.select('#grid')
	.append('div').attr('class', 'container');


// draw the images i.e. start everything
drawCalendar(calendar);


function drawCalendar(calendar) {
	// month container
	var row = container.selectAll('.row')
		.data(Object.entries(calendar)).enter()
		.append('div').attr('class', 'row')
		.append('div').attr('class', 'col-12');

	// title
	row.append('div').attr('class', 'row')
		.append('h3').attr('class', 'col').text((d) => d[0]);

	// day container
	var block = row.append('div').attr('class', 'row').selectAll('.day-cell')
		.data((d) => Object.entries(d[1])).enter()
			.append('div').attr('class', 'day-cell')
			.append('a').attr('class', 'day-block')
			.attr('href', (d) => BASE_URL + 'video/' + d[1].date).text((d) => d[0])
			.append('div')
		
	// get color scale for label count
	var color = d3.scaleLinear().domain([0, 0.1 + d3.max(block.data(), (d) => d[1].label_count)]).range(['black', 'MediumOrchid']).interpolate(d3.interpolateLab)
	
	// add badges to display image count and label count
	block.append('span').attr('class', 'badge badge-pill badge-dark')
		.text((d) => d[1].view_count + '/' + d[1].image_count + ' viewed')
	block.append('span').attr('class', 'badge badge-pill badge-dark')
		.text((d) => d[1].label_count + ' labels')
		.style('background', (d) => color(d[1].label_count))
	
	// block.append('span').attr('class', 'pct');

	// add on column sizes
	changeRowCount(1);
}

function changeRowCount(i){
	// var i = parseInt(this.value);
	container.selectAll('.day-cell')
		.attr('class', 'day-cell'
			 + ' col-lg-' + gridSizes[i]
			 + ' col-md-' + gridSizes[Math.min(i+1, gridSizes.length-1)]
			 + ' col-sm-' + gridSizes[Math.min(i+2, gridSizes.length-1)]
			 + ' col-'    + gridSizes[Math.min(i+3, gridSizes.length-1)])
}


