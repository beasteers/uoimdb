/*

Init

*/

var video_cursor = 0;
var video_data = null;
var video_interval = null;

// hash: filter, image index, video speed

// create image grid container
var row = d3.select('#grid');

var timeline = d3.select('#video-timeline');
timeline.append('div').attr('class', 'timeline'); // grey bar across entire screen
timeline.append('div').attr('class', 'tl-marker'); // blue progress bar
timeline.append('div').attr('class', 'tl-marker-current'); // darkgrey indicator for current img
timeline.append('div').attr('class', 'tl-marker-center')
	.append('span').attr('class', 'fa fa-caret-down'); // darkgrey indicator for center img

// bind control events
var nav = d3.select('.navbar');
var controls = nav.select('#controls');

var videoSpeed = controls.select('#videoSpeed')
	.on('change', throttle(function() { 
		set_hash(this.value, 2);
		if(window.video_interval) 
			play(); // restart interval with new speed
	}, 600)); 

set_hash(videoSpeed.node().value, 2);



function currentImage() {
	return video_data ? video_data[video_cursor] : {};
}

function selectCurrentImageContainer(){
	var nodes = row.selectAll('.image-container').nodes();
	return d3.select(nodes[nodes.length - 1]); // last image is the current image
}

function getBoxes() { // used in /save
	return video_data.reduce((result, d) => d.boxes ? result.concat(d.boxes) : result, []); 
}

window.img_meta = {};
function setImageMeta(key, value, message=false, i=null) {
	var src = video_data[i != null ? i : video_cursor].src;
	if(!img_meta[src]) {
		img_meta[src] = {}
	}
	img_meta[src][key] = value;
	if(message)
		displayMessage('Current image '+key+' set to "'+value+'".', 2000);
}




/*

Control Actions

*/

var play_pause = nav.select('#play-pause').on('click', togglePlay);

var rewind = nav.select('#rewind').on('click', function(){
	updateVideoPosition(config.REWIND_STEP);
});

var step_back = nav.select('#step-back').on('click', function(){
	updateVideoPosition(config.BACK_STEP);
});

var step_forward = nav.select('#step-forward').on('click', function(){
	updateVideoPosition(config.FORWARD_STEP);
});

var fastforward = nav.select('#fastforward').on('click', function(){
    updateVideoPosition(config.FASTFORWARD_STEP);
});

var mark_unreviewed = nav.select('#mark-unreviewed').on('click', function(){
	setImageMeta('status', 'unreviewed', true);
	var nodes = row.selectAll('.image-cell').nodes();
	d3.select(nodes[nodes.length - 1]).select('.status').text('unreviewed');
});


/*

Key Bindings

*/


d3.select('body').on('keydown.prev-page', function(){
    if(d3.event.shiftKey && d3.event.keyCode == config.KEYS.PREV_PAGE) {
    	d3.select('#prev_query').dispatch('click');
    	d3.event.stopImmediatePropagation();
    }
})
d3.select('body').on('keydown.next-page', function(){
    if(d3.event.shiftKey && d3.event.keyCode == config.KEYS.NEXT_PAGE) {
    	d3.select('#next_query').dispatch('click');
    	d3.event.stopImmediatePropagation();
    }
})

d3.select('body').on('keypress.play-pause', function(){
	if(d3.event.keyCode == config.KEYS.PLAY_PAUSE) 
		togglePlay();
})
d3.select('body').on('keydown.step-back', function(){
    if(d3.event.keyCode == config.KEYS.STEP_BACK) 
    	updateVideoPosition(config.BACK_STEP);
})
d3.select('body').on('keydown.step-forward', function(){
    if(d3.event.keyCode == config.KEYS.STEP_FORWARD) 
    	updateVideoPosition(config.FORWARD_STEP);
})

d3.select('body').on('keydown.mark-unreviewed', function(){
    if(d3.event.keyCode == config.KEYS.MARK_UNREVIEWED) 
    	nav.select('#mark-unreviewed').dispatch('click');
})


d3.select('body').on('keydown.draw-ghostboxes', function(){
    if(d3.event.keyCode == config.KEYS.DRAW_GHOSTBOXES){
		selectCurrentImageContainer().selectAll('.pt.ghost').dispatch('click');
    } 
})


d3.select('body').on('keydown.select-all', function(){
    if((d3.event.ctrlKey || d3.event.metaKey) && d3.event.keyCode == config.KEYS.SELECT_ALL) {
		var pts = selectCurrentImageContainer().selectAll('.pt:not(.ghost)');
		
		if(d3.event.shiftKey) { // if shift is pressed, toggle all
			pts.classed('selected', function(){ return !d3.select(this).classed('selected'); })
		} 
		else { // select all, or deselect if all are selected
			var all_selected = pts.size() == pts.filter('.selected').size();
			pts.classed('selected', !all_selected);
		}	

		d3.event.preventDefault();
	}
})

d3.select('body').on('keydown.delete', function(){
    if((d3.event.ctrlKey || d3.event.metaKey) && d3.event.keyCode == config.KEYS.DELETE_SELECTED)
		selectCurrentImageContainer().selectAll('.pt.selected').each(function(){
			d3.select(this).call(removeBox);
		});
})









/*

Timeline/image initialization

*/


function drawTimeline(data) {
	window.video_data = data.timeline; // globally accessible

	var dx = (1. / video_data.length)*100;
	timeline.select('.tl-marker-current').style('width', dx + '%');
	timeline.select('.tl-marker-center').style('width', dx + '%')
		.style('display', data.i_center != null ? 'block' : 'none')
		.style('left', data.i_center != null ? ((data.i_center / video_data.length)*100 + '%') : 0);

	d3.select('#prev_query.ajax').attr('href', data.prev_query).classed('d-none', data.prev_query);
	d3.select('#next_query.ajax').attr('href', data.next_query).classed('d-none', data.next_query);


	timeline.on('click', function(){
			pause();
			var mouse = d3.mouse(this);
			var box = this.getBoundingClientRect();
			var percent = mouse[0] / box.width;
			updateVideoPercent(percent);
		})
		.call(bindTooltip, tooltip)
		.on('mousemove.update-tooltip', function(){
			tooltip.html('')
			var current = video_data[video_cursor];
			if(!current) 
				tooltip.append('span').call(badge, 'dark').text(`current (${video_cursor}) ${current.date}`);

			var mouse = d3.mouse(this);
			var box = this.getBoundingClientRect()
			var percent = mouse[0] / box.width;
			var img = video_data[Math.floor(video_data.length * percent)];
			tooltip.append('span').call(badge, 'dark').text(`${img.date}`);
		}, true);


	video_data.forEach((d, i) => {d.index = i}); // add index so we know where to put the labels

	var label_markers = timeline.selectAll('.label-stack')
		.data(video_data.filter((d) => d.boxes));

	label_markers.enter()
		.append('div').attr('class', 'label-stack')
		.style('left', (d) => (d.index / video_data.length)*100 + '%') //  - dx/2.
		.style('width', dx + '%')
		.selectAll('.label-marker')
			.data((d) => d.boxes).enter()
			.append('div').attr('class', 'label-marker')
			.sort((a, b) => d3.ascending(config.LABELS.indexOf(a.label), config.LABELS.indexOf(b.label)))
			.style('background-color', getLabelColor)
			.call(bindTooltip, tooltip)
			.on('mousemove.update-tooltip', function(d){
				tooltip.append('span').call(badge, 'dark').text(`${d.label},${d.user}`);
			});

	label_markers.exit().remove();

	
	window.video_cursor = parseInt(get_hash()[1]) || 0;
	if(data.i_center != null) {
		window.video_cursor = data.i_center;
		setImageMeta('status', 'reviewed');
	}	

	set_hash(window.video_cursor, 1);
	updateVideoPosition(0);
}



/*

Video control

*/


function updateVideoPercent(percent) {
	updateVideoPosition(Math.floor(percent * video_data.length) - window.video_cursor);
}

window.frameRequested = false;
function updateVideoPosition(i, preload_n, preload_n_prev) {
	if(!video_data || !video_data.length) return;

	i = window.video_cursor + (i || 0);
	i = Math.min(Math.max(0, i), video_data.length - 1);

	// start loading image queue
	window.frameRequested = true;
	window.video_cursor = i;
	var imgs_data = video_data.slice(
		Math.max(i - (preload_n_prev != null ? preload_n_prev : config.PRELOAD_N_PREV_IMAGES), 0), 
		Math.min(i + (preload_n != null ? preload_n : config.PRELOAD_N_IMAGES), video_data.length)
	);
	preloadImages(imgs_data.map((d) => d.src));
	imageArrived(); // check if first image is already loaded
}

function imageArrived(src) {
	if(!video_data || src && src != video_data[window.video_cursor].src) return;

	var i = window.video_cursor;
	var img = window.imageQueue[video_data[i].src];
	
	var percent = 1.*i / video_data.length;
    marker = timeline.select('.tl-marker').datum({percent: percent})
        .style('width', percent*100 + '%').classed('buffering', true);
    timeline.select('.tl-marker-current').datum(img)
        .style('left', percent*100 + '%');

	if(!window.frameRequested || !img || !img.isLoaded)
		return; // not ready to change frames yet

	// set the marker position
	//var percent = 1.*i / video_data.length;
	//timeline.select('.tl-marker').datum({percent: percent})
	//	.style('width', percent*100 + '%');
	//timeline.select('.tl-marker-current').datum(img)
	//	.style('left', percent*100 + '%');
	marker.classed('buffering', false);

	// draw the images
	var imgs_data = video_data.slice(i, i + config.N_IMAGE_ELEMENTS);
	updateImages(imgs_data.reverse());
	row.selectAll('.image-cell img').on('click.pause', pause); // add pause on click
	set_hash(i, 1);
	window.frameRequested = false;
}




/*

Video control

*/

function play() {
	pause(); // if not already paused
	window.video_interval = setInterval(function() {
		if(window.video_cursor >= window.video_data.length-1){
			pause();
		} else if(!window.frameRequested) {
			updateVideoPosition(1);
		}

	}, parseInt(videoSpeed.node().value));
	play_pause.classed('fa-play', false).classed('fa-pause', true);
}

function pause() {
	if(window.video_interval){
		clearInterval(window.video_interval);
		window.video_interval = null;
		play_pause.classed('fa-pause', false).classed('fa-play', true);
	}
}

function togglePlay() {
	if(window.video_interval)
		pause();
	else
		play();
}





/*

Utilities

*/


Array.prototype.sortBy = function(func){
	return this.sort((a, b) => func(a) > func(b));
};

function throttle(fn, restPeriod){ 
	// only fire a function every so often
	var free = true;
	return function(){
		if (free){
			fn.apply(this, arguments);
			free = false;
			setTimeout((d) => { free = true; }, restPeriod);
		}
	}
}
