


function objectContain(el, selector) {
	var container = el.node();
	console.log(container);
	new ResizeSensor(container, function() {
		var outerAspect = container.clientWidth / container.clientHeight;
	    el.selectAll(selector).each(function(){
	    	var innerAspect = this.clientWidth / this.clientHeight;
	    	var aspects = innerAspect / outerAspect;
	    	d3.select(this)
	    		.attr('width', container.clientWidth * (aspects > 1 ? 1 : aspects) + 'px')
	    		.attr('height', container.clientHeight / (aspects < 1 ? 1 : aspects) + 'px')

	    })
	});
}




function get_init_cookie(key, def_value) {
	var value = Cookies.get(key);
	if(value === undefined && def_value !== undefined) {
		Cookies.set(key, def_value);
		return def_value;
	}
	return value;
}




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


function getLabelColor(d) {
	if(d.label) {
		var i = config.LABELS.indexOf(d.label);
		if(i < config.COLORS.length) {
			return config.COLORS[i];
		}
	}
	return config.DEFAULT_COLOR;
}


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

// function displayMessage(message, duration=5000, context='dark') {
// 	var id = 'message_'+uid();
// 	d3.select('#messages')
// 		.append('span')//.call(badge, context)
// 		.attr('id', id)
// 		.html(message)
// 		.style('opacity', 0).transition().duration(400).style('opacity', 1);

// 	// remove message after 6s
// 	setTimeout(function(){
// 		d3.select('#'+id)
// 			.transition().duration(400)
// 			.style('opacity', 0)
// 			.remove();
// 	}, duration);

// }




function printStatus(status) {
	if(!status) {
		return 'reviewing';
	}
	else if(status == 'reviewed'){
		return 'already reviewed.'
	}
	else {
		return status;
	}
}



/*

Hash utilities

*/

function get_hash() { return decodeURIComponent(window.location.hash.slice(1)).split('|') || new Array(4).fill(''); }
function set_hash(val, i) {
	if(i !== undefined) {
		hash = get_hash();
		hash[i] = val;
		hash = hash.join('|');
	}
	else {
		hash = val;
	}
	
	// window.location.hash = hash.join('|');
	history.replaceState(undefined, undefined, "#" + hash); // changes history instead of adding new entry
}





function proper_mod(v, n) {
    return ((v%n)+n)%n;
};


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
