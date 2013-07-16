
// auto-refresh rate
var tick = 200;
var autorefresh = true;

// show error in document (non-intrusive alert())
function showErr(err){
	document.getElementById("ErrorBox").innerHTML = err;
}

// show debug message in document
function msg(err){
	document.getElementById("MsgBox").innerHTML = err;
}

// called on change of auto-refresh button
function setautorefresh(){
	autorefresh =  document.getElementById("AutoRefresh").checked;
}

// Id of element that has focus. We don't auto-refresh a focused textbox
// as this would overwrite the users input.
var hasFocus = "";
function notifyfocus(id){hasFocus = id;}
function notifyblur (id){hasFocus = "";}

// onreadystatechange function for update http request.
// refreshes the DOM with new values received from server.
function refreshDOM(req){
	if (req.readyState == 4) { // DONE
		if (req.status == 200) {	
			showErr("");
			var response = JSON.parse(req.responseText);	
			for(var i=0; i<response.length; i++){
				var r = response[i];
				var elem = document.getElementById(r.ID);
				// switch element type
				if (elem.value != null && hasFocus != r.ID){ // textbox etc
					elem.value = r.HTML;
				}else{                                       // other elements
					elem.innerHTML = r.HTML;
					elem.value = r.HTML; // hack
				}
			}
		} else {
			showErr("Disconnected");	
		}
	}
}

// refreshes the contents of all dynamic elements.
// periodically called via setInterval()
function refresh(){
	if (autorefresh){
		try{
			var req = new XMLHttpRequest();
			req.open("POST", "/refresh/", true);
			req.timeout = tick;
			req.onreadystatechange = function(){ refreshDOM(req) };
			req.send(null);
		}catch(e){
			showErr(e); // TODO: same message as refresh
		}
	}
}

setInterval(refresh, tick);

// remote procedure call, called on button clicks etc.
function rpc(model, method, arg){
	try{
		var req = new XMLHttpRequest();
		req.open("POST", "/rpc/", false);
		var map = {"ID": model, "Method": method, "Arg": arg};
		req.send(JSON.stringify(map));
	}catch(e){
		showErr(e); // TODO
	}
	refresh();
}

function call(model){
	rpc(model, "call");
}

function settext(model){
	rpc(model, "set", document.getElementById("guielem_"+model).value);
}


