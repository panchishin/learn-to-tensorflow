awk '/^#/{ print $0 } /^([^#]|$)/{gsub("=.*","= # TODO",$0); gsub("sess.run.*","# TODO session execution command here",$0); print $0 }'
