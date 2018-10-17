
$.notify.defaults({ className: "success" });



$('.set-cookie').on('change', function() {
	name = this.name;
	value = this.value;
	Cookies.set(name, value);
	$.notify('Assigned '+name+' as '+value);
	console.log('Assigned cookie '+name+' as '+value, 'success');

	if($(this).data('remove'))
		d3.select(this).transition().style('opacity', 0).remove()
})





// $('#updateModalOptions').on('click', function() {
// 	for(var inp of $('#modalOptionsForm').serializeArray()) {
// 		Cookies.set(inp.name, inp.value);
// 	}
// })






