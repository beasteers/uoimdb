


function ImageLabelerApp(grid) {
	var app = this;
	this.grid = grid;

	Box.app = this;
	
	this.imageQueue = {};
	this.edited_data = {};
	this.image_meta = {};

	this.video_cursor = 0;
	this.video_data = null;
	this.video_interval = null;
	this._frameRequested = false;


	this.timeline = d3.select('#video-timeline');
	this.timeline.append('div').attr('class', 'timeline'); // grey bar across entire screen
	this.timeline.append('div').attr('class', 'tl-marker'); // blue progress bar
	this.timeline.append('div').attr('class', 'tl-marker-current'); // darkgrey indicator for current img
	this.timeline.append('div').attr('class', 'tl-marker-center')
		.append('span').attr('class', 'fa fa-caret-down'); // darkgrey indicator for center img

	this.tooltip = d3.select('#tooltip');



	/* 

	Actions

	*/

	var nav = d3.select('.navbar');
	var expand_nav = nav.select('#expand-nav').on('click', function(){
		nav.classed('collapsed', !nav.classed('collapsed')); // toggle collapsed
	});

	var current_image_filter = get_init_cookie('image_filter', config.DEFAULT_FILTER);

	var previous_filter = current_image_filter == 'Original' ? img_filters[0] : null; 
	this.toggle_original = nav.select('#toggle-original').on('click', function(){
		if(previous_filter){
			app.changeImageFilter(previous_filter);
			previous_filter = null;
		}
		else {
	        previous_filter = app.image_filter.node().value;	
			app.changeImageFilter('Original');
		}
	});

	this.play_pause = nav.select('#play-pause').on('click', () => this.togglePlay());

	this.rewind = nav.select('#rewind').on('click', function(){
		app.updateVideoPosition(config.REWIND_STEP);
	});

	this.step_back = nav.select('#step-back').on('click', function(){
		app.updateVideoPosition(config.BACK_STEP);
	});

	this.step_forward = nav.select('#step-forward').on('click', function(){
		app.updateVideoPosition(config.FORWARD_STEP);
	});

	this.fastforward = nav.select('#fastforward').on('click', function(){
	    app.updateVideoPosition(config.FASTFORWARD_STEP);
	});

	this.mark_unreviewed = nav.select('#mark-unreviewed').on('click', function(){
		app.setImageMeta('status', '', true);
		var nodes = app.grid.selectAll('.image-cell').nodes();
		d3.select(nodes[nodes.length - 1]).select('.status').text('unreviewed');
	});



	/* 

	Controls

	*/

	// bind control events
	// this.controls = d3.select('#controls');
	this.controls = d3.select('#nav');

	var video_speed = get_init_cookie('video_speed', config.SPEED.DEFAULT);
	
	this.videoSpeed = this.controls.select('#videoSpeed')
		.attr('value', video_speed)
		.on('change', throttle(function() { 
			Cookies.set('video_speed', this.value);
			if(app.video_interval) 
				app.play(); // restart interval with new speed
		}, 800)); 


	this.image_filter = d3.select('.image_filter') // change image filter
		.on('change', function(){ app.changeImageFilter(this.value) })
		.selectAll('option')
		.data(img_filters).enter()
		.append('option')
		.attr('value', (d) => d)
		.text((d) => d)
		.property('selected', (d) => d == current_image_filter);

	this.save_button = this.controls.select('#save').on('click', () => this.saveBoxes()); // save label locations


	d3.selectAll('.load-query').on('click', function(){
		var href = d3.select(this).attr('href')
		if(href && href != '#') {
			if(d3.select(this).classed('ajax')) {
				d3.event.preventDefault();
				app.loadImagesFromQuery(href);
			} 
			else {
				window.location = href;
			}
		}
	});



	/* 

	Keyboard Shortcuts

	*/

	d3.select('body').on('keypress.change-label', function(){
		var key = d3.event.keyCode - 48; // 48=0, 57=9
		if(0 <= key && key <= 9 && key < config.LABELS.length) {
			key = proper_mod(key - 1, 10); // shift so pressing 1 gets 0th label, 0 gets 10th
			var label = config.LABELS[key];
			if(label)
				d3.selectAll('.pt.selected .label-selection select').property('value', label).dispatch('change');
		}
	});

	d3.select('body').on('keydown.save', function(){
		if((d3.event.metaKey || d3.event.ctrlKey) && d3.event.keyCode == config.KEYS.SAVE) { // 83=s
			app.save_button.dispatch('click');
			d3.event.preventDefault();
		}
	});

	d3.select('body').on('keydown.toggle-original', function(){
	    if(d3.event.keyCode == config.KEYS.TOGGLE_ORIGINAL) 
	    	app.toggle_original.dispatch('click'); // up arrow
	})

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
			app.togglePlay();
	})
	d3.select('body').on('keydown.step-back', function(){
	    if(d3.event.keyCode == config.KEYS.STEP_BACK) 
	    	app.updateVideoPosition(config.BACK_STEP);
	})
	d3.select('body').on('keydown.step-forward', function(){
	    if(d3.event.keyCode == config.KEYS.STEP_FORWARD) 
	    	app.updateVideoPosition(config.FORWARD_STEP);
	})

	d3.select('body').on('keydown.mark-unreviewed', function(){
	    if(d3.event.keyCode == config.KEYS.MARK_UNREVIEWED) 
	    	nav.select('#mark-unreviewed').dispatch('click');
	})


	d3.select('body').on('keydown.draw-ghostboxes', function(){
	    if(d3.event.keyCode == config.KEYS.DRAW_GHOSTBOXES){
			app.selectCurrentImageContainer().selectAll('.pt.ghost').dispatch('click');
	    } 
	})


	d3.select('body').on('keydown.select-all', function(){
	    if((d3.event.ctrlKey || d3.event.metaKey) && d3.event.keyCode == config.KEYS.SELECT_ALL) {
			var pts = app.selectCurrentImageContainer().selectAll('.pt:not(.ghost)');
			
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
			app.selectCurrentImageContainer().selectAll('.pt.selected').each(function(){
				d3.select(this).call(Box.remove);
			});
	})



	/* 

	Autosave / Save Warning

	*/

	// warn about unsaved changes before closing.
	// or just autosave!!
	window.addEventListener("beforeunload", function (e) {
		var n_labels = Object.keys(app.edited_data).length;
		var n_metas = Object.keys(app.image_meta).length;
		console.log(n_labels, n_metas);
	    if (!n_labels && !n_metas) return;

	    if(config.AUTOSAVE) {
	    	app.saveBoxes();
	    }
	    else {
	    	var confirmationMessage = 'You have ' + n_labels + ' unsaved labels. '
		                            + 'If you leave before saving, your changes will be lost.';

		    (e || window.event).returnValue = confirmationMessage; //Gecko + IE
		    return confirmationMessage; //Gecko + Webkit, Safari, Chrome etc.
	    }
	});

	return this;
}



ImageLabelerApp.prototype.selectCurrentImageContainer = function(){
	var nodes = this.grid.selectAll('.image-container').nodes();
	return d3.select(nodes[nodes.length - 1]); // last image is the current image
}



/*

Timeline/image initialization

*/


ImageLabelerApp.prototype.drawTimeline = function(data) {
	var app = this;
	this.video_data = data.timeline; // globally accessible
	this.video_data.forEach((d, i) => {d.index = i}); // add index so we know where to put the labels

	// set timeline cursor
	var dx = (1. / this.video_data.length)*100;
	this.timeline.select('.tl-marker-current').style('width', dx + '%');
	this.timeline.select('.tl-marker-center').style('width', dx + '%')
		.style('display', data.i_focus != null ? 'block' : 'none')
		.style('left', data.i_focus != null ? ((data.i_focus / this.video_data.length)*100 + '%') : 0);

	// add previous/next button queries
	d3.select('#prev_query.ajax').attr('href', data.prev_query).classed('d-none', data.prev_query);
	d3.select('#next_query.ajax').attr('href', data.next_query).classed('d-none', data.next_query);

	// timeline click and tooltip
	this.timeline.on('click', function(){
			app.pause();
			app.updateVideoPercent(d3.mouse(this)[0] / this.getBoundingClientRect().width);
		})
		.call(bindTooltip, this.tooltip)
		.on('mousemove.update-tooltip', function(){
			app.tooltip.html('')
			var current = app.video_data[app.video_cursor];
			if(!current) 
				app.tooltip.append('span').call(badge, 'dark').text(`current (${app.video_cursor}) ${current.date}`);

			var percent = d3.mouse(this)[0] / this.getBoundingClientRect().width;
			var img = app.video_data[Math.floor(app.video_data.length * percent)];
			app.tooltip.append('span').call(badge, 'dark').text(`${img.date}`);
		}, true);


	
	// draw label indicators
	var label_markers = this.timeline.selectAll('.label-stack')
		.data(this.video_data.filter((d) => d.boxes));

	label_markers.enter()
		.append('div').attr('class', 'label-stack')
		.style('left', (d) => (d.index / this.video_data.length)*100 + '%') //  - dx/2.
		.style('width', dx + '%')
		.selectAll('.label-marker')
			.data((d) => d.boxes).enter()
			.append('div').attr('class', 'label-marker')
			.sort((a, b) => d3.ascending(config.LABELS.indexOf(a.label), config.LABELS.indexOf(b.label)))
			.style('background-color', getLabelColor)
			.call(bindTooltip, this.tooltip)
			.on('mousemove.update-tooltip', function(d){
				app.tooltip.append('span').call(badge, 'dark').text(`${d.label},${d.user}`);
			});

	label_markers.exit().remove(); 

	
	this.video_cursor = parseInt(get_hash()[1]) || 0;
	if(data.i_focus != null) {
		this.video_cursor = data.i_focus;
		if(!this.video_data[this.video_cursor].status) {
			this.setImageMeta('status', 'reviewed');
		}
	}	

	set_hash(this.video_cursor);
	this.updateVideoPosition(0);
}









ImageLabelerApp.prototype.loadImagesFromQuery = function(query) {
	var app = this;
	console.log('Getting images from: ', query);
	$.get( query )
	.done(function(data) {
		console.log(data);
		app.drawTimeline(data);
	}).fail(function(data){
		console.log(data);
	});
}



ImageLabelerApp.prototype.preloadImages = function(images) {
	var filter = this.controls.select('.image_filter').node().value; // get the current image filter
	var app = this;

	// preload image queue
	this.imageQueue = images.reduce(function(o, src){
		var url = BASE_URL + 'filter/' + filter + '/' + src;

		if(app.imageQueue[src]) {
			o[src] = app.imageQueue[src];
		}
		else {
			var img = new Image();
			img.isLoaded = false;
			img.onload = function(){
				this.isLoaded = true;
				app.imageArrived(src); // update once the current image is loaded
			};
			img.src = url + (config.CACHE_ALL_IMAGES ? '?cache_result=1' : '');
			o[src] = img;
		}
		return o;
	}, {});
} 


ImageLabelerApp.prototype.imageArrived = function(src) {
	if(!this.video_data || src && src != this.video_data[this.video_cursor].src) return;

	var i = this.video_cursor;
	var img = this.imageQueue[this.video_data[i].src];
	
	var percent = 1.*i / this.video_data.length;
    
    marker = this.timeline.select('.tl-marker').datum({percent: percent})
        .style('width', percent*100 + '%').classed('buffering', true);
    
    this.timeline.select('.tl-marker-current').datum(img)
        .style('left', percent*100 + '%');

	if(!this._frameRequested || !img || !img.isLoaded)
		return; // not ready to change frames yet

	marker.classed('buffering', false);

	// draw the images
	var imgs_data = this.video_data.slice(i, i + config.N_IMAGE_ELEMENTS);
	this.updateImages(imgs_data.reverse());
	this.grid.selectAll('.image-cell img').on('click.pause', () => this.pause()); // add pause on click
	set_hash(i);
	this._frameRequested = false;
}


ImageLabelerApp.prototype.updateImages = function(images) {
    if(!images) return;

    // bind image data
    var cell = this.grid.selectAll('.image-cell').data(images, (d) => d.src).order();
   
    // remove old images
    cell.exit().remove();

	// deselect previously selected points
	cell.selectAll('.pt').classed('selected', false);
   
    // add wrapper for image and points
    cell_new = cell.enter().append('div').attr('class', 'image-cell').attr('data-src', (d) => d.src); // the bootstrap column cell

    // add title
    cell_new.append('div').attr('class', 'title')
        .html((d) => `${d.src} -- <b>${d.date}<b>`);// -- <span class="status">${printStatus(d.status)}</span>

    // draw image and boxes
    var images = cell_new
    	// .append('div').attr('class', 'image-container-wrap-y')
    	// .append('div').attr('class', 'image-container-wrap-x')
    	.append('div').attr('class', 'image-container') // a wrapper to contain image + annotations
        .append((d) => this.imageQueue[d.src])
		.attr('draggable', 'false')
        .on('click.create-box', function(d){
            d3.select(this).call(Box.create);
        }).each(function(d, i){
            // draw already saved bounding boxes
            var container = d3.select(this.parentNode);
            var boxes = (d.boxes || []).filter((d) => d.src);
            boxes.forEach((b) => container.call(Box.draw, b));
        });

    // cell_new.selectAll('.image-container-wrap').each(function(){ d3.select(this).call(objectContain, 'img') })

    // draw ghost boxes
    this.grid.selectAll('.image-cell .image-container img').each(function(d){
    	var container = d3.select(this.parentNode);
    	container.selectAll('.pt.ghost').remove();

    	// draw bounding boxes from previous
    	if(d.index && d.index == app.video_cursor){
        	var prev_ids = (d.boxes || []).filter((b) => b.src && b.prev_id).map((b) => b.prev_id);
        	var prev_boxes = (app.video_data[d.index - 1].boxes || []).filter((b) => b.src && !prev_ids.includes(b.id));
            prev_boxes.forEach((b) => 
            	container.call(Box.drawGhost, Object.assign({}, b, {src: d.src, id: uid(), prev_id: b.id})));
        }
    })
}






ImageLabelerApp.prototype.updateVideoPercent = function(percent) {
	this.updateVideoPosition(Math.floor(percent * this.video_data.length) - this.video_cursor);
}

ImageLabelerApp.prototype.updateVideoPosition = function(i, preload_n, preload_n_prev) {
	if(!this.video_data || !this.video_data.length) return;

	i = this.video_cursor + (i || 0);
	i = Math.min(Math.max(0, i), this.video_data.length - 1);

	// start loading image queue
	this._frameRequested = true;
	this.video_cursor = i;
	var imgs_data = this.video_data.slice(
		Math.max(i - (preload_n_prev != null ? preload_n_prev : config.PRELOAD_N_PREV_IMAGES), 0), 
		Math.min(i + (preload_n != null ? preload_n : config.PRELOAD_N_IMAGES), this.video_data.length)
	);
	this.preloadImages(imgs_data.map((d) => d.src));
	this.imageArrived(); // check if first image is already loaded
}




ImageLabelerApp.prototype.play = function() {
	var app = this;
	this.pause(); // if not already paused
	this.video_interval = setInterval(function() {
		if(app.video_cursor >= app.video_data.length-1){
			app.pause();
		} else if(!app.frameRequested) {
			app.updateVideoPosition(1);
		}

	}, parseInt(this.videoSpeed.node().value));
	this.play_pause.select('i').classed('fa-play', false).classed('fa-pause', true);
}

ImageLabelerApp.prototype.pause = function() {
	if(this.video_interval){
		clearInterval(this.video_interval);
		this.video_interval = null;
		this.play_pause.select('i').classed('fa-pause', false).classed('fa-play', true);
	}
}

ImageLabelerApp.prototype.togglePlay = function() {
	return this.video_interval ? this.pause() : this.play();
}















ImageLabelerApp.prototype.changeImageFilter = function(filter){
	this.controls.select('.image_filter').node().value = filter;
	this.grid.selectAll('.image-cell img').attr('src', (d) => BASE_URL + 'filter/' + filter + '/' + d.src);
	Cookies.set('image_filter', filter);
	this.imageQueue = {}
	//preloadImages()
}



ImageLabelerApp.prototype.setImageMeta = function(key, value, message=false, i=null) {
	var src = this.video_data[i != null ? i : this.video_cursor].src;
	if(!this.image_meta[src]) {
		this.image_meta[src] = {}
	}
	this.image_meta[src][key] = value;
	if(message)
		$.notify('Current image '+key+' set to "'+value+'".')
}

ImageLabelerApp.prototype.markBoxEdited = function(data) {
	this.edited_data[data.id] = data;
}

ImageLabelerApp.prototype.saveBoxes = function(){
	var boxes = Object.values(this.edited_data); // getBoxes();
	var app = this;

	$.post( BASE_URL + 'save/', {
			boxes: JSON.stringify(boxes),
			sample_name: sample_name,
			img_meta: JSON.stringify(app.image_meta)
		})
		.done(function(data) {
			console.log(data);
			$.notify(data.message);
			app.edited_data = {};
		})
		.fail(function(data){
			$.notify('error saving..', 'error');
			console.log(data);
		});
}






