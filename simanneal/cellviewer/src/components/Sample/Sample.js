import React, { Component } from 'react';
import Slides from '../Slides/Slides.js';
import $ from 'jquery';
import './Sample.css';
import srcTree from "../../output/test.svg";
import srcPie from "../../output/pie.svg";

export default class Sample extends Component {
    constructor(props) {
        super(props);

        this.state={
            tog: false
        }

        this.read = this.read.bind(this);
        this.images = []
        for(var i=0; i<=50; i++){
            var x = i.toString();
            while (x.length !==3){
                x = "0"+x;
            }
            this.images.push([
                require('../../output/frame'+x+'.png'),
                'frame'+x+'.png'
            ]);
        }
        this.colony=null;

        
    }

    componentDidMount(){
        $.ajax({
            // url: '../src/output/colony.csv',
            url: 'colony.csv',
            context: document.body
        }).done(this.read);
    }

    read(file){
        this.colony={};
        file = file.split('\n');
        for(var i=1; i<file.length; i++){
            var data = file[i].split(",");
            data[0] = data[0].split("/").pop();
            data[2] = Number(data[2]);
            data[3] = Number(data[3]);
            if(!this.colony.hasOwnProperty(data[0])){ 
                this.colony[data[0]] = [0,100]
            }
            this.colony[data[0]].push([data[1],data[2],data[3]])
            if(this.colony[data[0]][0]< data[1].length-1){
                this.colony[data[0]][0] = data[1].length-1;
            }
            if(this.colony[data[0]][1]> data[1].length-1){
                this.colony[data[0]][1] = data[1].length-1;
            }
            
        }
        this.setState({
            tog:!this.state.tog
        });
    }

    render() {
        if(this.colony===null){
            return null;
        }
        return (
            <Slides 
                imgs={this.images} 
                srcTree={srcTree}
                src_pie={srcPie}
                colony={this.colony}/>
        );
    }
} 
