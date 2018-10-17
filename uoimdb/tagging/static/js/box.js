window.Box = {

	create: function(element) {
		var node = element.node(); // image
		var parent = node.parentNode; // image container
		var d = element.datum();

        var mouse = d3.mouse(node);
        var bbox = parent.getBoundingClientRect();

        var x = mouse[0] / bbox.width,
        	y = mouse[1] / bbox.height;
        
        // draw a new bounding box
        d3.select(parent).call(Box.draw, {
            src: d.src, id: uid(), 
            label: config.LABELS[0],
            x: x, y: y, w: 0, h: 0,
        });

        if(!d3.event.shiftKey) {
			d3.selectAll('.pt').classed('selected', false);
		}
    },

	draw: function(element, data, selected){
		//if(!d3.event.shiftKey) 
		//	d3.selectAll('.pt').classed('selected', false);

		// draws a new bounding box
		var pt = element
			.append('div').attr('class', 'pt').data([data])
			.classed('removed', (d) => d.src == false)
			.classed('selected', selected)
			.call(Box.update, data)
			.call(Box.store);
		

		pt.on('click.select-box', Box.select)
			.call(d3.drag().on('drag', Box.dragStart).on('end', Box.dragEnd));

		pt.append('div').attr('class', 'label-handle')
			.append('div').attr('class', 'label-selection')
			.append('select')
				.on('change', function(){ pt.call(Box.update, null, {label: this.value}).call(Box.store); })
				.selectAll('option').data(config.LABELS).enter()
				.append('option')
				.attr('value', (d) => d)
				.text((d) => d)
				.property('selected', (d) => d == data.label);

		pt.append('span').attr('class', 'box-credit')
			.text((d) => d.user);
	},

	drawGhost: function(element, data) {
		var pt = element.data([data])
			.append('div').attr('class', 'pt ghost')
			.call(Box.update, data);

		pt.on('click.create-box', function(d){
			d3.select(this.parentNode).call(Box.draw, d, true);
			if(Box.app)
				Box.app.markBoxEdited(d);
			d3.select(this).remove();
		})
	},

	select: function(){
		var selected = d3.select(this).classed('selected');
		if(!d3.event.shiftKey) {
			d3.selectAll('.pt').classed('selected', false);
		}
		d3.select(this).classed('selected', !selected);
	},


	dragStart: function(data){
		var bbox = this.parentNode.getBoundingClientRect();
		var lbox = this.getBoundingClientRect();

		if(d3.event.sourceEvent.shiftKey) {
			var w, h;
			// resize the bounding box
			var mouse = d3.mouse(this);
			w = Math.abs(mouse[0]*2 - lbox.width)
			h = Math.abs(mouse[1]*2 - lbox.height)
			if(config.FIXED_ASPECT_RATIO) {
				var dim = Math.max(w, h);
				w = dim;
				h = dim;
			}

			var new_data = {
				w: w / bbox.width,
				h: h / bbox.height
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
		
		d3.select(this).call(Box.update, data, new_data).call(Box.store);
	},

	dragEnd: function(d){
		// remove points when dragged out of bounds
		if(Box.outOfBounds(d)) {
			d3.select(this).call(Box.remove);
		}
	},


	/**/

	update: function(element, data, new_data){
		if(element.empty()) return;
		data = data || element.datum();
		if(new_data) {
			data = Object.assign(data, new_data);
			if(Box.app)
				Box.app.markBoxEdited(data);
		}

		// var bbox = element.node().parentNode.getBoundingClientRect();
		element.data([data])
			.style('width', (d) => 100. * d.w + '%')
			.style('height', (d) => 100. * d.h + '%')
			.style('left', (d) => 100. * (d.x - d.w/2) +'%')
			.style('top', (d) => 100. * (d.y - d.h/2) +'%')
			.style('border-color', getLabelColor)
			.classed('hover-out', (d) => Box.outOfBounds(d));
	},

	
	store: function(element) {
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
	},

	remove: function(element) {
		var data = element.datum();
		element.datum(Object.assign(data, {src: false}))
	           .classed('removed', true);
		if(Box.app) {
			Box.app.markBoxEdited(data);
		}
	},

	outOfBounds: function (pt){
		return pt.x < 0 || pt.x > 1 || pt.y < 0 || pt.y > 1;
	}
}









