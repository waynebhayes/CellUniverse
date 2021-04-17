import React, { Component } from 'react';
import { Tooltip } from 'reactstrap';
import { SketchPicker } from 'react-color';
import { Button } from '@material-ui/core';

export default class Cell extends Component {
    constructor(props) {
        super(props);
        this.toggle = this.toggle.bind(this);
        this.colorChange = this.colorChange.bind(this);
        this.save = this.save.bind(this);
        this.state = {
            tooltipOpen: false,
            clicked: 0,
            background: props.color
        };
        this.temp_color = props.color;
    }

    toggle(n) {
        this.setState({
            tooltipOpen: !this.state.tooltipOpen
            // ,background: "#FFFFFF"
          });
    }

    click(n){
        this.setState({ clicked: n });
    }

    colorChange(color){
        this.temp_color = color.hex;
    }

    save(){
        this.props.colors[this.props.k.slice(1)] = this.temp_color;
        this.setState({
            background: this.temp_color
        });
    }
    colorRevert(color){
        this.i = this.props.k.length-1;
        while(!this.props.colors.hasOwnProperty(this.props.k.slice(1,this.i))){
            this.i--;
        }
        this.props.colors[this.props.k.slice(1)] = this.props.colors[this.props.k.slice(1,this.i)];
        this.setState({
            background: this.props.colors[this.props.k.slice(1)]
        })
    }



    render() {
        this.picker = 
                <SketchPicker
                    color={ this.temp_color }
                    onChangeComplete={ this.colorChange }
                />

        return (
            <div>
                <div 
                    id={"Tooltip"+this.props.k}
                    onClick={()=>this.click(1)}
                    style={{
                        position:"absolute",
                        top:this.props.top,
                        left:this.props.left,
                        backgroundColor:this.state.background,
                        width:"7px",
                        height:"7px"
                    }}/>
                <div 
                    id={"TooltipR"+this.props.k}
                    onClick={()=>this.click(2)}
                    style={{
                        position:"absolute",
                        bottom:this.props.bottomR,
                        left:this.props.leftR-3.5,
                        backgroundColor:this.state.background,
                        width:"7px",
                        height:"7px"
                    }}/>
                <Tooltip placement="right" isOpen={this.state.tooltipOpen} target={"Tooltip"+this.props.k} toggle={()=>this.toggle(0)}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"yellow",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.props.k}
                    </div>
                </Tooltip>
                <Tooltip placement="right" isOpen={this.state.clicked===1} target={"Tooltip"+this.props.k}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"#A1E9F1",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.picker}

                        <Button onClick={()=>{this.click(0);}} style={{width:"33%"}}>
                            Cancel
                        </Button>
                        <Button onClick={()=>{this.click(0);this.save();}} style={{width:"33%"}}>
                            Set
                        </Button>
                        <Button onClick={()=>{this.click(0);this.colorRevert();}} style={{width:"33%"}}>
                            Revert
                        </Button>
                    </div>
                </Tooltip>
                <Tooltip placement="right" isOpen={this.state.clicked===2} target={"TooltipR"+this.props.k}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"#A1E9F1",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.picker}
                        <Button onClick={()=>{this.click(0);}} style={{width:"33%"}}>
                            Cancel
                        </Button>
                        <Button onClick={()=>{this.click(0);this.save();}} style={{width:"33%"}}>
                            Set
                        </Button>
                        <Button onClick={()=>{this.click(0);this.colorRevert();}} style={{width:"33%"}}>
                            Revert
                        </Button>
                    </div>
                </Tooltip>
                <Tooltip placement="right" isOpen={this.state.tooltipOpen} target={"TooltipR"+this.props.k} toggle={()=>this.toggle(1)}>
                    <div 
                        style={{
                            marginLeft:"2px",
                            padding: "1px",
                            textAlign:"center",
                            backgroundColor:"yellow",
                            borderTopLeftRadius:"0px",
                            borderBottomLeftRadius:"0px",
                            borderTopRightRadius:"5px",
                            borderBottomRightRadius:"5px"}}>
                        {this.props.k}
                    </div>
                </Tooltip>
                
            </div>
        );
    }
}
