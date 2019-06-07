import React, { Component} from 'react';
import upload from "./upload.png";
import Slides from '../Slides/Slides.js';
import angles from '../../output/angle.json';
import srcTree from "../../output/tree.svg";
import srcPie from "../../output/pie.svg";
import colony from "../../output/colony.json";

export default class Viewer extends Component {
    constructor(props) {
        super(props);
        this.state = {
            progress: 1000
        };
        this.updateInputValue = this.updateInputValue.bind(this);
        this.readCol = this.readCol.bind(this);
        
        this.data0 = {
            angles: angles,
            colony: colony,
            imgs: [],
            srcTree: srcTree,
            src_pie: srcPie
        }

        this.data = {
            angles: angles,
            colony: colony,
            imgs: [],
            srcTree: srcTree,
            src_pie: srcPie
        }

        for(var i=0; i<=50; i++){
            var x = i.toString();
            while (x.length !==3){
                x = "0"+x;
            }
            this.data.imgs.push(require('../../output/frame'+x+'.png'));
        }

        this.l = 1000;

        this.hash = {};
    }

    getImage(i,p){
        var reader = new FileReader();
        reader.onload = (e) => {
            this.data.imgs[i] = e.target.result;
            this.setState({
                progress: this.state.progress+1
            });
        };
        reader.readAsDataURL(this.hash[p]);
    }

    readCol(e){
        this.data.colony = JSON.parse(e.target.result);
        this.setState({
            progress: this.state.progress+1
        });
        this.l = this.data.colony["frames"]+4;

        var anglereader = new FileReader();
        anglereader.onload = (e) => {
            this.data["angles"] = JSON.parse(e.target.result);
            this.setState({
                progress: this.state.progress+1
            });
        };
        anglereader.readAsText(this.hash["angle.json"]);

        var piereader = new FileReader();
        piereader.onload = (e) => {
            this.data["src_pie"] = e.target.result;
            this.setState({
                progress: this.state.progress+1
            });
        };
        piereader.readAsDataURL(this.hash["pie.svg"]);

        var treereader = new FileReader();
        treereader.onload = (e) => {
            this.data["srcTree"] = e.target.result;
            this.setState({
                progress: this.state.progress+1
            });
        };
        treereader.readAsDataURL(this.hash["tree.svg"]);

        for(var i=0; i<this.l-4; i++){
            this.getImage(i,this.data.colony[i.toString()]["frame"].split("/").pop());
        }
    }

    updateInputValue(e) {

        if (e.target.files.length !== 0) {
            this.l = 1000;
            this.hash = {};
            this.data = {
                angles: {},
                colony: {},
                imgs: [],
                srcTree: null,
                src_pie: null
            }
            var files = e.target.files;
            for(var i=0; i<files.length; i++){
                this.hash[files[i].name] = files[i];
            }
            var colreader = new FileReader();
            colreader.onload = this.readCol;
            colreader.readAsText(this.hash["colony.json"]);

            this.setState({
                progress: 0
            });
        } 
    }

    render() {
        var display = 1;
        if(this.state.progress >= this.l){
            display = 0;
            for(var key in this.data0){
                this.data0[key] = this.data[key];
            }
            this.data0 = this.data;
            this.slides = <Slides 
                    id = "slide"
                    {...this.data0}
            />;
        }
        var progress = this.state.progress/this.l;
        return (
            <div>
                {this.slides}
                <img
                src={upload} alt=""
                style={{
                    position:"absolute",
                    top:"2%",
                    right:"2%",
                    width:"4%",
                    height:"7%"
                }} />
                <input
                    style={{
                        position:"absolute",
                        top:"2%",
                        right:"2%",
                        width:"4%",
                        height:"7%",
                        opacity:"0"
                    }}
                    directory="" webkitdirectory=""
                    type="file" name="file" id="File"
                    onChange={evt => this.updateInputValue(evt)}
                />
                <div style={{
                    display:["none","initial"][display],
                    position:"absolute",
                    top:"45vh",
                    left:"25vw",
                    width:"50vw",
                    height:"10vh"
                }}>
                    <div
                        style={{
                            border: "3px solid blue",
                            borderRadius:"20px",
                            backgroundColor:"transparent",
                            width:progress.toString()+"%",
                            height:"3vh"
                        }}
                    />
                    <br/>
                    <div style={{color:"white",textAlign:"center"}}>{progress.toString().split(".")[0]}%</div>
                </div>
            </div>
        );
    }
}

