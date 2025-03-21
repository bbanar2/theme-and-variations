inlets = 1
outlets = 2

function send_notes(){
	// Get selected clip slot
	var api = new LiveAPI()
	api.path = 'live_set'
	
	// TODO
	// is_arrangement_clip
	
	// Get tempo
	tempo = api.get('tempo')
	
	// Get selected clip
	api.path = 'live_set view'
	clip_id = parseInt(api.get('highlighted_clip_slot')[1])
	detail_clip_id = parseInt(api.get('detail_clip')[1])
	api.id = detail_clip_id
	// Get looping details
	looping = api.get('looping')
	
	generating = 1;
	abort_generation = false;
	
	data = get_notes(clip_id)
	
	outlet(0, data['notes'])
	
}



function get_notes(id){
	var api = new LiveAPI()
	api.id = id
	
	if (api.type === 'ClipSlot'){
		api.id = api.get('clip')[1]
	}
	
	n_notes = api.call('get_notes', 0, 0, 1600, 128)[1]
	n_selected_notes = 0
	
	var offset = 0;
	notes = api.call('get_notes', 0, 0, 16, 128).slice(0, -1)
	n_selected_notes += notes[1]
	while (n_selected_notes < n_notes) {
		offset += 1
		new_notes = api.call('get_notes', offset * 16, 0, 16, 128)
		n_selected_notes += new_notes[1]
		notes = notes.concat(new_notes.slice(2, -1))
	}

	notes[1] = n_notes
	max_idx = n_notes * 6 + 2
	start = notes[4]
	end = notes[4]
	duration = notes[5]
	
	for (var i = 4; i < max_idx; i = i+6){
		if (notes[i] > end){
			end = notes[i]
			duration = notes[i+1]
		}
		if (notes[i] < start){
			start = notes[i]
		}
	}
	
	buffer = notes
	
	// Get looping details
	looping = api.get('looping')
	loop_start = api.get('loop_start')[0]
	loop_end = api.get('loop_end')[0]
	
	return {
		id : api.id,
		notes : notes,
		clip_start : start,
		clip_end : Math.max(end+duration, loop_end),
		selected_region : {
	  					   'start' : loop_start,
						   'end' : loop_end,
						  }
	}
}

function write_note(clip_id, pitch, time, duration, velocity, muted){
	var api = new LiveAPI()
	var note;
	api.id = clip_id
	api.id = parseInt(api.get('clip')[1])

	api.call('set_notes')
	api.call("notes", 1);
	api.call("note", pitch, time.toFixed(4), duration.toFixed(4), velocity, muted);
	api.call("done");
}