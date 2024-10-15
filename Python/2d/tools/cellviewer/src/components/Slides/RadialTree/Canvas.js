import React, { Component } from 'react';
import { Stage, Layer, Line, Path } from 'react-konva';

export default class Canvas extends Component {

    constructor(props){
        super(props);

        this.nodes = {};
        this.maxlayer = 1;
        Object.keys(props.frames).forEach( 
            (name) =>{
                if(name!=="name"){
                    this.nodes[name] = {
                        "name": name,
                        "angle": props.angles[name],
                        "start_frame": props.frames[name][0],
                        "end_frame": props.frames[name][1],
                    }
                    if(props.angles.hasOwnProperty(name+'0') && props.angles.hasOwnProperty(name+'1')){
                        this.nodes[name]["children"] = [props.angles[name+'0'],props.angles[name+'1']]
                    }
                    this.maxlayer = Math.max(this.maxlayer, props.frames[name][1])
                }
            }
        )
        this.step_size = (props.l/(this.maxlayer+2)/2)*0.8;
    }

    draw_line(node){
        var line = null;
        var line2 = null;
        var c=node.name.length;
        var curr = this.props.curr + 1;
        if(curr >= node.end_frame){
            while(c>1 && !this.props.colors.hasOwnProperty(node.name.substring(1,c))) c-=1
            line = <Line
                x={window.innerWidth/4}
                y={window.innerHeight/2}
                key={node.name+"line"}
                points={[this.step_size*node.start_frame, 0,this.step_size*node.end_frame, 0]}
                stroke={this.props.colors[node.name.substring(1,c)]}
                rotation={-180*node.angle/Math.PI}
            />;
        } else if(this.props.posC){
            line2 = <Line
                x={window.innerWidth/4}
                y={window.innerHeight/2}
                key={node.name+"line2"}
                points={[this.step_size*node.start_frame, 0,this.step_size*node.end_frame, 0]}
                stroke="gray"
                rotation={-180*node.angle/Math.PI}
            />;
        }
        
        if(curr < node.end_frame && curr >= node.start_frame){
            while(c>1 && !this.props.colors.hasOwnProperty(node.name.substring(1,c))) c-=1
            line = <Line
                x={window.innerWidth/4}
                y={window.innerHeight/2}
                key={node.name+"line"}
                points={[this.step_size*node.start_frame, 0,this.step_size*curr, 0]}
                stroke={this.props.colors[node.name.substring(1,c)]}
                rotation={-180*node.angle/Math.PI}
            />;
        }
        console.log(line2)
        return <>{line2}{line}</>;
    }

    draw_arc(node){
        var path = null;
        var path2 = null;
        if(node.children){
            var max_angle = node.children[0]
            var min_angle = node.children[1]
            if(this.props.curr >= node.end_frame-1){
                var c=node.name.length;
                while(c>1 && !this.props.colors.hasOwnProperty(node.name.substring(1,c))) c-=1

                path = <Path
                    key={node.name+"arc"}
                    x={window.innerWidth/4}
                    y={window.innerHeight/2}
                    stroke={this.props.colors[node.name.substring(1,c)]}
                    data= { 'M'+this.step_size*node.end_frame*Math.cos(-max_angle)+
                            ','+this.step_size*node.end_frame*Math.sin(-max_angle)+
                            'A'+this.step_size*node.end_frame+
                            ','+this.step_size*node.end_frame+
                            ',0,0,1'+
                            ','+this.step_size*node.end_frame*Math.cos(-min_angle)+
                            ','+this.step_size*node.end_frame*Math.sin(-min_angle)}
                />
            }else if(this.props.posC){
                path2 = <Path
                    key={node.name+"arc"}
                    x={window.innerWidth/4}
                    y={window.innerHeight/2}
                    stroke="gray"
                    data= { 'M'+this.step_size*node.end_frame*Math.cos(-max_angle)+
                            ','+this.step_size*node.end_frame*Math.sin(-max_angle)+
                            'A'+this.step_size*node.end_frame+
                            ','+this.step_size*node.end_frame+
                            ',0,0,1'+
                            ','+this.step_size*node.end_frame*Math.cos(-min_angle)+
                            ','+this.step_size*node.end_frame*Math.sin(-min_angle)}
                />
            }
        }
        return <>{path2}{path}</>;
    }

    render() {

        this.step_size = (this.props.l/(this.maxlayer+2)/2)*0.8;
        return (
        <Stage width={window.innerWidth/2} height={window.innerHeight}
            style={{
                position:"absolute",
                right: 0,
                top: 0
            }}>
            <Layer>
                {Object.keys(this.nodes).map(
                    (node) => this.draw_line(this.nodes[node])
                )}
                {Object.keys(this.nodes).map(
                    (node) => this.draw_arc(this.nodes[node])
                )}
            </Layer>
        </Stage>
        );
    }
}
