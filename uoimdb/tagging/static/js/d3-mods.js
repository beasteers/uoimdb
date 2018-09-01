d3.selection.prototype.first = function() {
  return d3.select(this[0][0]);
};


d3.selection.prototype.last = function() {
  return d3.select(this[0][this.size() - 1]);
};


d3.selection.prototype.moveToFront = function() {  
  return this.each(function(){
    this.parentNode.appendChild(this);
  });
};


d3.selection.prototype.moveToBack = function() {  
    return this.each(function() { 
        var firstChild = this.parentNode.firstChild; 
        if (firstChild) { 
            this.parentNode.insertBefore(this, firstChild); 
        } 
    });
};


d3.selection.prototype.closest = function(selector){ 
    var matches = [];
    this.each(function(){
        var el = this;
        while(el){
            el = el.parentNode;
            if(el && el.matches(selector)){
                matches.push(el);
                return;
            }
        }
    });
    return d3.selectAll(matches);
}

d3.selection.prototype.parent = function(){ 
    return d3.select(this.parentNode);
}


