

function loadImagesFromQuery(query) {
	console.log('Getting images from: ', query);
	$.get( query )
	.done(function(data) {
		console.log(data);
		drawTimeline(data);
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

function bindTooltip(element, tooltip) {
	element.on('mouseover.bind-tooltip', function(){
			tooltip.classed('visible', true).html('')
				.style('left', d3.event.pageX+'px')
				.style('top', d3.event.pageY+'px')
		})
		.on('mousemove.bind-tooltip', function(){
			tooltip.style('top', d3.event.pageY+'px').style('left', d3.event.pageX+'px');
		})
		.on('mouseout.bind-tooltip', function(){
			tooltip.classed('visible', false);
		});
}




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


function proper_mod(v, n) {
    return ((v%n)+n)%n;
};

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
			img.src = url;
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
    cell_new = cell.enter().append('div').attr('class', 'image-cell'); // the bootstrap column cell

    // add title
    cell_new.append('div').attr('class', 'title')
        .html((d) => `${d.src} -- <b>${d.date}<b>`);

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




/*

Boxes
 - draw box
 - update box parameters
 - mark box as removed
 - mark box as edited
 - store box in box list
 - draw ghost box

*/


function drawGhostBox(element, data) {
	var pt = element.data([data])
		.append('div').attr('class', 'pt ghost')
		.call(updateBox, data);

	pt.on('click.create-box', function(d){
		d3.select(this.parentNode).call(drawBox, d, true);
		markBoxEdited(d);
		d3.select(this).remove();
	})
}


function drawBox(element, data, selected){
	//if(!d3.event.shiftKey) 
	//	d3.selectAll('.pt').classed('selected', false);

	// draws a new bounding box
	var pt = element
		.append('div').attr('class', 'pt').data([data])
		.classed('removed', (d) => d.src == false)
		.classed('selected', selected)
		.call(updateBox, data)
		.call(storeBox);
	

	pt.on('click.select-box', function(){
			var selected = d3.select(this).classed('selected');
			if(!d3.event.shiftKey) {
				d3.selectAll('.pt').classed('selected', false);
			}
			d3.select(this).classed('selected', !selected);
		})
		.call(d3.drag().on('drag', function(data){
			var bbox = this.parentNode.getBoundingClientRect();

			if(d3.event.sourceEvent.shiftKey) {
				// resize the bounding box
				var mouse = d3.mouse(this);
				// var dim = Math.max(Math.abs(mouse[0]), Math.abs(mouse[1]));
				// data.w = dim / bbox.width;
				// data.h = dim / bbox.height;	
				var new_data = {
					w: Math.abs(mouse[0]) / bbox.width,
					h: Math.abs(mouse[1]) / bbox.height
				};
			}
			else {
				// drag center of box
				var mouse = d3.mouse(this.parentNode);
				var new_data = {
                    x: mouse[0] / bbox.width,
                    y: mouse[1] / bbox.height
                };
			}
			
			d3.select(this).call(updateBox, data, new_data).call(storeBox);
			
		})
		.on('end', function(d){
			// remove points when dragged out of bounds
			if(outOfBounds(d)) {
				d3.select(this).call(removeBox);
			}
		}));
	pt.append('div').attr('class', 'label-handle')
		.append('div').attr('class', 'label-selection')
		.append('select')
			.on('change', function(){
				pt.call(updateBox, null, {label: this.value}).call(storeBox);
			})
			.selectAll('option').data(config.LABELS).enter()
			.append('option')
			.attr('value', (d) => d)
			.text((d) => d)
			.property('selected', (d) => d == data.label);

	pt.append('span').attr('class', 'box-credit')
		.text((d) => d.user);
}


function updateBox(element, data, new_data){
	if(element.empty()) return;
	data = data || element.datum();
	if(new_data) {
		data = Object.assign(data, new_data);
		markBoxEdited(data);
	}

	// var bbox = element.node().parentNode.getBoundingClientRect();
	element.data([data])
		.style('width', (d) => 100. * d.w + '%')
		.style('height', (d) => 100. * d.h + '%')
		.style('left', (d) => 100. * (d.x - d.w/2) +'%')
		.style('top', (d) => 100. * (d.y - d.h/2) +'%')
		.style('border-color', getLabelColor)
		.classed('hover-out', (d) => outOfBounds(d));
}

function removeBox(element) {
	var data = element.datum();
	element.datum(Object.assign(data, {src: false}))
           .classed('removed', true);
	markBoxEdited(data);
}

window.edited_data = {};
function markBoxEdited(data) {
	window.edited_data[data.id] = data;
}




function storeBox(element) {
	var d = element.datum();
	var cell = element.closest('.image-cell');
	var data = cell.datum();
	data.boxes = data.boxes || []; // cuz empty is stored as a string -.- (pandas+json was being dumb)
	var i = data.boxes.map((d) => d.id).indexOf(d.id);
	if(i > -1) {
		data.boxes[i] = d;
	}
	else {
		data.boxes.push(d);
	}
	cell.datum(data);
	//console.log(data.boxes.length, i, data.boxes, d)
}

function outOfBounds(pt){
	return pt.x < 0 || pt.x > 1 || pt.y < 0 || pt.y > 1;
}

function getLabelColor(d) {
	if(d.label) {
		var i = config.LABELS.indexOf(d.label);
		if(i < config.COLORS.length) {
			return config.COLORS[i];
		}
	}
	return config.DEFAULT_COLOR;
}


function changeImageFilter(filter){
	controls.select('#imgFilter').node().value = filter;
	row.selectAll('.image-cell img')
		.attr('src', (d) => BASE_URL + 'filter/' + filter + '/' + d.src);
	set_hash(filter, 0);
	window.imageQueue = {}
	//preloadImages()
}






/*

Save boxes

*/

function getBoxes(){
	return row.selectAll('.pt:not(.ghost)').data();
}

function saveBoxes(){
	var boxes = Object.values(window.edited_data); // getBoxes();
	console.log(boxes);

	// d3.json('/save', {
	// 	method: 'POST', 
	// 	body: JSON.stringify({boxes: data}), 
	// 	headers: {"Content-Type": "application/json"},
	// })
	// .then(function(data) {
	// 	console.log(data);
	// });



	$.post( BASE_URL + 'save/', {
		boxes: JSON.stringify(boxes),
		sample_name: sample_name,
		img_meta: JSON.stringify(img_meta)
	})
	.done(function(data) {
		console.log(data);
		displayMessage('Saved &#128077;');
		window.edited_data = {};
	});
}

// function saveImageMeta(){
// 	$.post( BASE_URL + 'save/meta/')
// 	.done(function(data) {
// 		console.log(data);
// 		displayMessage('Saved metadata &#128077;');
// 	});
// }

// function saveEverything(){
// 	saveImageMeta();
// 	saveBoxes();
// }

// warn about unsaved changes before closing.
window.addEventListener("beforeunload", function (e) {
	var n_labels = Object.keys(window.edited_data).length;
	var n_metas = Object.keys(window.img_meta).length;
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






/* 

Display messags
*/


function _randomLetter() {
	return String.fromCharCode(65 + Math.floor(Math.random() * 26));
}

function uid(){
	// generates a unique id as long as it's not fired too fast
	//return (new Date()).getTime();
	return _randomLetter() + _randomLetter() + Date.now();
}

function badge(el, context='dark'){
	return el.attr('class', 'badge badge-pill badge-' + context);
}

function displayMessage(message, duration=5000, context='dark') {
	var id = 'message_'+uid();
	d3.select('#messages')
		.append('span').call(badge, context)
		.attr('id', id)
		.html(message)
		.style('opacity', 0).transition().duration(400).style('opacity', 1);

	// remove message after 6s
	setTimeout(function(){
		d3.select('#'+id)
			.transition().duration(400)
			.style('opacity', 0)
			.remove();
	}, duration);

}



/*

Hash utilities

*/

function get_hash() { return decodeURIComponent(window.location.hash.slice(1)).split('|') || new Array(4).fill(''); }
function set_hash(val, i) {
	hash = get_hash();
	hash[i] = val;
	// window.location.hash = hash.join('|');
	history.replaceState(undefined, undefined, "#" + hash.join('|')); // changes history instead of adding new entry
}


