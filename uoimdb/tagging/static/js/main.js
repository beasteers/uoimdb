

function loadImagesFromQuery(query) {
	console.log('Getting images from: ', query);
	$.get( query )
	.done(function(data) {
		console.log(5);
		console.log(data);
		drawTimeline(data);
	}).fail(function(data){
		console.log(data);
	});
}

d3.selectAll('.load-query').on('click', function(){
	var href = d3.select(this).attr('href')
	if(href && href != '#') {
		if(d3.select(this).classed('ajax')) {
			d3.event.preventDefault();
			loadImagesFromQuery(href);
		} 
		else {
			window.location = href;
		}
	}
	
});

/*

Tooltip

*/

var tooltip = d3.select('#tooltip')






var nav = d3.select('.navbar');
var expand_nav = nav.select('#expand-nav').on('click', function(){
	nav.classed('collapsed', !nav.classed('collapsed')); // toggle collapsed
});


var previous_filter = get_hash()[0] == 'Original' ? img_filters[0] : null; 
var toggle_original = nav.select('#toggle-original').on('click', function(){
	if(previous_filter){
		changeImageFilter(previous_filter);
		previous_filter = null;
	}
	else {
        previous_filter = controls.select('#imgFilter').node().value;	
		changeImageFilter('Original');
	}
});


// bind control events
var controls = d3.select('#controls');

controls.select('#save').on('click', saveBoxes); // save label locations

controls.select('#imgFilter') // change image filter
	.on('change', function(){changeImageFilter(this.value)})
	.selectAll('option')
	.data(img_filters).enter()
	.append('option')
	.attr('value', (d) => d)
	.text((d) => d)
	.property('selected', (d) => d == get_hash()[0]);




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
		controls.select('#save').dispatch('click');
		d3.event.preventDefault();
	}
});
d3.select('body').on('keydown.toggle-original', function(){
    if(d3.event.keyCode == config.KEYS.TOGGLE_ORIGINAL) toggle_original.dispatch('click'); // up arrow
}) 





window.imageQueue = {};
function preloadImages(images) {
	var filter = controls.select('#imgFilter').node().value; // get the current image filter

	// preload image queue
	window.imageQueue = images.reduce(function(o, src){
		var url = BASE_URL + 'filter/' + filter + '/' + src;

		if(window.imageQueue[src]) {
			o[src] = window.imageQueue[src];
		}
		else {
			var img = new Image();
			img.isLoaded = false;
			img.onload = function(){
				this.isLoaded = true;
				imageArrived(src); // update once the current image is loaded
			};
			img.src = url + (config.CACHE_ALL_IMAGES ? '?cache_result=1' : '');
			o[src] = img;
		}
		return o;
	}, {});
} 

function imageArrived(src) {} // by default do nothing


function updateImages(images) {
    if(!images) return;

    // bind image data
    var cell = row.selectAll('.image-cell').data(images, (d) => d.src).order();
   
    // remove old images
    cell.exit().remove();

	// deselect previously selected points
	cell.selectAll('.pt').classed('selected', false);
   
    // add wrapper for image and points
    cell_new = cell.enter().append('div').attr('class', 'image-cell').attr('data-src', (d) => d.src); // the bootstrap column cell

    // add title
    cell_new.append('div').attr('class', 'title')
        .html((d) => `${d.src} -- <b>${d.date}<b>`);// -- <span class="status">${printStatus(d.status)}</span>

    // draw image
    var images = cell_new.append('div').attr('class', 'image-container') // a wrapper to contain image + annotations
        .append((d) => window.imageQueue[d.src])
		.attr('draggable', 'false')
        .on('click.create-box', function(d){
            var mouse = d3.mouse(this);
            var bbox = this.parentNode.getBoundingClientRect();

            var x = mouse[0] / bbox.width,
            	y = mouse[1] / bbox.height;
            
            // draw a new bounding box
            d3.select(this.parentNode).call(drawBox, {
                src: d.src, id: uid(), label: config.LABELS[0],
                x: x, y: y, w: 0, h: 0,
            });

            if(!d3.event.shiftKey) {
				d3.selectAll('.pt').classed('selected', false);
			}
        }).each(function(d, i){
            // draw already saved bounding boxes
            if(d.boxes){
            	var el = d3.select(this.parentNode);
                d.boxes.forEach(function(b){
                    if(d.src)
                        el.call(drawBox, b, i==0);
                });
            }
        });

    row.selectAll('.image-cell .image-container img').each(function(d){
    	var prev_ids = (d.boxes || []).filter((d) => d.src && d.prev_id).map((d) => d.prev_id);

    	var el = d3.select(this.parentNode);
    	el.selectAll('.pt.ghost').remove()

    	// draw bounding boxes from previous
    	if(d.index && d.index == video_cursor){
        	var d_prev = window.video_data[d.index - 1];
        	if(d_prev.boxes){
        		
        		d_prev.boxes.forEach(function(b){
                    if(b.src && !prev_ids.includes(b.id))
                        el.call(drawGhostBox, Object.assign({}, b, {src: d.src, id: uid(), prev_id: b.id}));
                });
        	}
        }
    })
}







// warn about unsaved changes before closing.
window.addEventListener("beforeunload", function (e) {
	var n_labels = Object.keys(window.edited_data).length;
	var n_metas = Object.keys(window.img_meta).length;
	console.log(n_labels, n_metas);
    if (!n_labels && !n_metas) return;

    if(config.AUTOSAVE) {
    	saveBoxes();
    }
    else {
    	var confirmationMessage = 'You have ' + n_labels + ' unsaved labels. '
	                            + 'If you leave before saving, your changes will be lost.';

	    (e || window.event).returnValue = confirmationMessage; //Gecko + IE
	    return confirmationMessage; //Gecko + Webkit, Safari, Chrome etc.
    }
});








